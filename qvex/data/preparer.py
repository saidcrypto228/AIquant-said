"""
qvex/data/preparer.py
========================
Robust data preparation for BTCUSDT 1m OHLCV data.
Handles local zip detection, Binance Vision downloads, and Hyperliquid bridging.
"""

import sys
import os
import zipfile
import time
import requests
import numpy as np
import pandas as pd
import logging
import re
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
RAW_DIR  = DATA_DIR / 'raw'
OUT_PATH = RAW_DIR / 'BTCUSDT_1m_full.parquet'

BINANCE_BASE = 'https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/'
CSV_COLS = ['open_time','open','high','low','close','volume',
            'close_time','quote_vol','n_trades','taker_buy_base','taker_buy_quote','ignore']
MONTHLY_PAT = re.compile(r'^BTCUSDT-1m-\d{4}-\d{2}\.zip$')

def _parse_csv(path: Path) -> pd.DataFrame:
    """Load a Binance Vision monthly CSV into a clean OHLCV DataFrame."""
    df = pd.read_csv(path, header=None, names=CSV_COLS, dtype=str)
    ts = df['open_time'].astype(np.int64)
    if ts.iloc[0] > 1_000_000_000_000_000:   # microseconds → milliseconds
        ts = ts // 1000
    df.index = pd.to_datetime(ts, unit='ms', utc=True)
    df.index.name = 'open_time'
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['open', 'high', 'low', 'close', 'volume']]

def _find_local_zips() -> list:
    """Search data/ and data/raw/ for bundle zip files."""
    zips = []
    for d in [DATA_DIR, RAW_DIR]:
        if not d.exists(): continue
        for z in sorted(d.glob('*.zip')):
            if MONTHLY_PAT.match(z.name): continue
            if z.stat().st_size < 100_000: continue
            if not zipfile.is_zipfile(z): continue
            zips.append(z)
    seen = set()
    result = []
    for z in zips:
        rp = z.resolve()
        if rp not in seen:
            seen.add(rp)
            result.append(z)
    return result

def _extract_zip(zip_path: Path) -> int:
    """Extract a zip into RAW_DIR. Handles flat and nested formats."""
    extracted = 0
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if '__MACOSX' in member or member.startswith('.'): continue
            name = Path(member).name
            if name.endswith('.csv') and 'BTCUSDT-1m-' in name:
                dest = RAW_DIR / name
                if not dest.exists():
                    with z.open(member) as src, open(dest, 'wb') as dst:
                        dst.write(src.read())
                    extracted += 1
                else:
                    extracted += 1
            elif name.endswith('.zip') and 'BTCUSDT-1m-' in name:
                inner_dest = RAW_DIR / name
                with z.open(member) as src, open(inner_dest, 'wb') as dst:
                    dst.write(src.read())
                with zipfile.ZipFile(inner_dest) as iz:
                    for inner_member in iz.namelist():
                        inner_name = Path(inner_member).name
                        if inner_name.endswith('.csv'):
                            csv_dest = RAW_DIR / inner_name
                            if not csv_dest.exists():
                                with iz.open(inner_member) as isrc, open(csv_dest, 'wb') as idst:
                                    idst.write(isrc.read())
                                extracted += 1
                            else:
                                extracted += 1
                inner_dest.unlink()
    return extracted

def _download_month(ym: str) -> bool:
    """Download BTCUSDT-1m-YYYY-MM.zip from Binance Vision and extract CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RAW_DIR / f'BTCUSDT-1m-{ym}.csv'
    if csv_path.exists():
        return True
    zip_url  = BINANCE_BASE + f'BTCUSDT-1m-{ym}.zip'
    zip_path = RAW_DIR / f'BTCUSDT-1m-{ym}.zip'
    try:
        r = requests.get(zip_url, timeout=120, stream=True)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW_DIR)
        zip_path.unlink()
        return True
    except Exception as e:
        logger.warning(f"Failed to download {ym}: {e}")
        if zip_path.exists(): zip_path.unlink()
        return False

def _fetch_hyperliquid() -> pd.DataFrame | None:
    """Fetch last 7 days from Hyperliquid to bridge gap to today."""
    try:
        url     = 'https://api.hyperliquid.xyz/info'
        now_ms  = int(time.time() * 1000)
        start_ms = now_ms - 7 * 24 * 60 * 60 * 1000
        r = requests.post(url, json={
            'type': 'candleSnapshot',
            'req': {'coin': 'BTC', 'interval': '1m',
                    'startTime': start_ms, 'endTime': now_ms}
        }, timeout=20)
        r.raise_for_status()
        bars = r.json()
        if not bars: return None
        rows = []
        for b in bars:
            ts_ms = int(b.get('t', b.get('time', b.get('T', 0))))
            rows.append({'open': float(b.get('o', 0)), 'high': float(b.get('h', 0)),
                         'low':  float(b.get('l', 0)), 'close': float(b.get('c', 0)),
                         'volume': float(b.get('v', 0)), 'open_time': ts_ms})
        hl = pd.DataFrame(rows)
        hl.index = pd.to_datetime(hl['open_time'], unit='ms', utc=True)
        hl.index.name = 'open_time'
        return hl[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        logger.warning(f"Hyperliquid bridge unavailable: {e}")
        return None

def ensure_data_prepared(days: int = 1825, verbose: bool = True):
    """
    Ensure the BTCUSDT 1m dataset is prepared and up to date.
    Checks for local zips, downloads from Binance Vision, and bridges with Hyperliquid.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Check if we already have a reasonably fresh parquet
    if OUT_PATH.exists():
        mtime = OUT_PATH.stat().st_mtime
        age_h = (time.time() - mtime) / 3600
        if age_h < 12: # If less than 12h old, assume it's fresh enough
            # We still might want to check if it has enough days
            try:
                df_meta = pd.read_parquet(OUT_PATH, columns=[])
                if len(df_meta) >= (days * 1400 * 0.9): # Rough check for bar count
                    if verbose: logger.info(f"Using existing dataset ({age_h:.1f}h old)")
                    return
            except Exception:
                pass

    if verbose:
        print(f"\n  AIQuant — Auto-Data Preparation  (target: {days} days)")
        print(f"  {'─' * 58}")

    dfs = []
    
    # 2. Local Zips
    local_zips = _find_local_zips()
    if local_zips:
        if verbose: print(f"  ✓ Found {len(local_zips)} local zip(s) — extracting...")
        for zp in local_zips:
            n = _extract_zip(zp)
            if verbose: print(f"    → {n} CSVs from {zp.name}")
    
    # 3. Binance Vision Download
    # Always check if we need more months based on 'days'
    try:
        from dateutil.relativedelta import relativedelta
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'python-dateutil'], check=True)
        from dateutil.relativedelta import relativedelta

    now_utc = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    start_date = now_utc - relativedelta(days=days)
    months = []
    cursor = start_date.replace(day=1)
    while cursor < now_utc:
        months.append(cursor.strftime('%Y-%m'))
        cursor += relativedelta(months=1)

    # Check which months we already have as CSVs
    existing_csvs = {f.name for f in RAW_DIR.glob('BTCUSDT-1m-????-??.csv')}
    needed_months = [m for m in months if f'BTCUSDT-1m-{m}.csv' not in existing_csvs]
    
    if needed_months:
        if verbose: print(f"  ↓ Downloading {len(needed_months)} missing months from Binance Vision...")
        for ym in needed_months:
            success = _download_month(ym)
            if verbose and success: print(f"    ✓ {ym}", end='\r')
        if verbose: print("\n  ✓ Binance Vision download complete")

    # 4. Load CSVs
    csv_files = sorted(RAW_DIR.glob('BTCUSDT-1m-????-??.csv'))
    if verbose: print(f"  ⚙ Loading {len(csv_files)} CSV files...")
    for f in csv_files:
        try:
            dfs.append(_parse_csv(f))
        except Exception as e:
            logger.warning(f"Failed to parse {f.name}: {e}")

    if not dfs:
        raise RuntimeError("No data found! Please ensure internet access or place BTCUSDT zips in data/ folder.")

    # 5. Hyperliquid Bridge
    if verbose: print(f"  ⚙ Fetching Hyperliquid bridge (last 7 days)...")
    hl = _fetch_hyperliquid()
    if hl is not None:
        dfs.append(hl)

    # 6. Combine and Save
    if verbose: print(f"  ⚙ Combining and saving to parquet...")
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep='last')]
    combined = combined.dropna()
    combined = combined[(combined['close'] > 0) & (combined['volume'] >= 0)]
    
    # Trim to requested days
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days + 1) # +1 for buffer
    combined = combined[combined.index >= cutoff]
    
    combined.to_parquet(OUT_PATH)
    if verbose:
        print(f"  ✓ Dataset prepared: {len(combined):,} bars")
        print(f"  ✓ Saved to {OUT_PATH.relative_to(BASE_DIR)}\n")

if __name__ == "__main__":
    # Test call
    ensure_data_prepared(days=30)