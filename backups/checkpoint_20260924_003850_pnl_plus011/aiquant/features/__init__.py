"""
aiquant/features/__init__.py
=============================
Feature engineering pipeline with verbose, timed progress output.

Memory strategy for large datasets (>1M bars):
- Technical and Microstructure stages run on the full DataFrame in-place
  (no copies, pd.concat with dict pattern)
- StatArb stage is chunked: processes CHUNK_SIZE bars at a time with OVERLAP
  overlap to avoid edge effects on rolling windows, then stitches results
- After each stage, gc.collect() is called to free intermediate memory
"""

import gc
import time
import platform
import warnings
import numpy as np

from .technical      import generate_all_technical_features
from .microstructure import generate_all_microstructure_features
from .statarb        import generate_all_statarb_features


# ── Colour helpers (no-op on Windows) ────────────────────────────────────────
def _c(code, text):
    if platform.system() == 'Windows':
        return text
    return f'\033[{code}m{text}\033[0m'

_GREEN  = lambda t: _c('32', t)
_CYAN   = lambda t: _c('36', t)
_YELLOW = lambda t: _c('33', t)
_DIM    = lambda t: _c('2',  t)
_BOLD   = lambda t: _c('1',  t)


def _step(label: str, fn, df, bar_count: int, verbose: bool = True):
    """Run a feature stage, print timing and column delta."""
    cols_before = df.shape[1]
    t0 = time.perf_counter()
    df = fn(df)
    elapsed = time.perf_counter() - t0
    cols_added = df.shape[1] - cols_before
    speed = bar_count / elapsed if elapsed > 0 else 0
    if verbose:
        print(
            f"    {_GREEN('✓')} {label:<22} "
            f"{_CYAN(f'+{cols_added} features'):>18}  "
            f"{_DIM(f'{elapsed:.1f}s')}  "
            f"{_DIM(f'{speed:,.0f} bars/s')}"
        )
    return df


def _statarb_chunked(df, verbose: bool = True):
    """
    Run StatArb feature engineering in memory-efficient chunks.

    For datasets > 1M bars, pd.concat on the full DataFrame would temporarily
    hold two copies (old + new columns) in RAM. Chunking avoids this by
    processing CHUNK_SIZE rows at a time with OVERLAP overlap, then stitching
    only the new columns back together.

    OVERLAP must be >= the largest rolling window used in statarb (240 bars).
    """
    import pandas as pd

    CHUNK_SIZE = 500_000   # rows per chunk (~350 MB at 171 cols float32)
    OVERLAP    = 500       # must be >= max rolling window (240 bars)
    n          = len(df)

    if n <= CHUNK_SIZE:
        # Small dataset — run directly, no chunking needed
        return generate_all_statarb_features(df)

    # ── Chunked path ──────────────────────────────────────────────────────────
    # We need to know which NEW columns statarb adds. Run on a tiny probe first.
    probe     = df.iloc[:1000].copy()
    probe_out = generate_all_statarb_features(probe)
    new_cols  = [c for c in probe_out.columns if c not in df.columns]
    del probe, probe_out
    gc.collect()

    # Pre-allocate output arrays for new columns.
    # String/object columns are identified by name — do NOT rely on probe dtype
    # inference because rolling-window columns may be all-NaN in the probe rows,
    # causing pandas to infer float64 even for string columns like 'regime'.
    # Any column whose name appears in KNOWN_STRING_COLS gets an object array;
    # everything else gets float32.
    KNOWN_STRING_COLS = {'regime', 'regime_label', 'market_state', 'trend_state'}

    new_col_arrays = {}
    for col in new_cols:
        if col in KNOWN_STRING_COLS:
            new_col_arrays[col] = np.full(n, None, dtype=object)
        else:
            new_col_arrays[col] = np.full(n, np.nan, dtype=np.float32)

    chunk_starts = list(range(0, n, CHUNK_SIZE))
    n_chunks     = len(chunk_starts)

    for ci, start in enumerate(chunk_starts):
        end         = min(start + CHUNK_SIZE, n)
        chunk_start = max(0, start - OVERLAP)   # include overlap from previous chunk
        chunk_end   = end

        chunk    = df.iloc[chunk_start:chunk_end].copy()
        chunk_out = generate_all_statarb_features(chunk)

        # Copy only the valid (non-overlap) rows back to output arrays
        valid_start_in_chunk = start - chunk_start   # skip the overlap rows
        valid_rows_in_chunk  = end - start

        for col in new_cols:
            if col in chunk_out.columns:
                arr = chunk_out[col].to_numpy()
                new_col_arrays[col][start:end] = arr[valid_start_in_chunk:valid_start_in_chunk + valid_rows_in_chunk]

        del chunk, chunk_out
        gc.collect()

        if verbose:
            pct = (ci + 1) / n_chunks * 100
            print(f"    {_DIM(f'  StatArb chunk {ci+1}/{n_chunks}  ({pct:.0f}%)')}", end='\r')

    if verbose:
        print(' ' * 60, end='\r')   # clear progress line

    # Attach new columns to the original DataFrame via pd.concat (one shot)
    import pandas as pd
    new_df = pd.DataFrame(new_col_arrays, index=df.index)
    result = pd.concat([df, new_df], axis=1)
    del new_col_arrays, new_df
    gc.collect()
    return result


def build_full_feature_set(df, verbose: bool = True) -> "pd.DataFrame":
    """
    Apply all feature groups with verbose per-step timing.

    Stages
    ------
    1. Technical    — trend, momentum, volatility, volume, candle  (~92 features)
    2. Microstructure — OFI, VPIN, Amihud, Kyle's λ, Roll spread  (~55 features)
    3. StatArb      — Hurst, half-life, ADF, Kalman, CUSUM         (~19 features)

    Performance
    -----------
    All hot-path rolling windows use Numba JIT-compiled C code (parallel=True).
    The ADF step uses a fast Numba approximation (~500x vs statsmodels loop).
    Call aiquant.utils.fast_math.warmup() once at startup to pre-compile JIT.

    Memory
    ------
    StatArb is chunked for datasets > 1M bars to avoid OOM on Colab (12 GB RAM).
    """
    import pandas as pd

    n_bars  = len(df)
    t_total = time.perf_counter()

    if verbose:
        print(f"\n  {_BOLD('Feature Engineering')}  ·  {n_bars:,} bars  ·  {df.shape[1]} input columns")
        print(f"  {'─' * 62}")

    # ── Numba warm-up (no-op if already compiled) ─────────────────────────
    if verbose:
        print(f"    {_CYAN('⚙')}  Numba JIT warm-up ...", end='\r')
    try:
        from ..utils.fast_math import warmup
        warmup()
    except Exception:
        pass
    if verbose:
        print(f"    {_GREEN('✓')} {'Numba JIT warm-up':<22} {'(cached)':>18}  {_DIM('ready')}")

    # ── Stage 1: Technical ────────────────────────────────────────────────
    if verbose:
        print(f"    {_CYAN('⚙')}  Technical indicators ...", end='\r')
    # Suppress the TSI divide-by-zero RuntimeWarning — it is handled by np.where
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning,
                                message='invalid value encountered in divide')
        df = _step("Technical indicators", generate_all_technical_features, df, n_bars, verbose)
    gc.collect()

    # ── Stage 2: Microstructure ───────────────────────────────────────────
    if verbose:
        print(f"    {_CYAN('⚙')}  Microstructure ...", end='\r')
    df = _step("Microstructure", generate_all_microstructure_features, df, n_bars, verbose)
    gc.collect()

    # ── Stage 3: StatArb (chunked for large datasets) ─────────────────────
    if verbose:
        print(f"    {_CYAN('⚙')}  StatArb / regime ...", end='\r')
    cols_before = df.shape[1]
    t0 = time.perf_counter()
    df = _statarb_chunked(df, verbose=verbose)
    elapsed    = time.perf_counter() - t0
    cols_added = df.shape[1] - cols_before
    speed      = n_bars / elapsed if elapsed > 0 else 0
    if verbose:
        print(
            f"    {_GREEN('✓')} {'StatArb / regime':<22} "
            f"{_CYAN(f'+{cols_added} features'):>18}  "
            f"{_DIM(f'{elapsed:.1f}s')}  "
            f"{_DIM(f'{speed:,.0f} bars/s')}"
        )
    gc.collect()

    # ── Drop NaN warmup rows ──────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)

    total_elapsed = time.perf_counter() - t_total

    if verbose:
        print(f"  {'─' * 62}")
        print(
            f"  {_GREEN('✓')} Done  ·  "
            f"{_BOLD(str(df.shape[1]))} features  ·  "
            f"{_BOLD(f'{len(df):,}')} usable bars  "
            f"{_DIM(f'({n_dropped} warmup rows dropped)')}"
        )
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        speed  = n_bars / total_elapsed if total_elapsed > 0 else 0
        print(
            f"  {_DIM('Memory:')} {mem_mb:.1f} MB  ·  "
            f"{_DIM('Total time:')} {total_elapsed:.1f}s  ·  "
            f"{_DIM('Throughput:')} {speed:,.0f} bars/s\n"
        )

    return df


__all__ = [
    'generate_all_technical_features',
    'generate_all_microstructure_features',
    'generate_all_statarb_features',
    'build_full_feature_set',
]
