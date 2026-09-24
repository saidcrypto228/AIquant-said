"""
aiquant/features/gpu_features.py
==================================
GPU-accelerated feature engineering using CuPy (CUDA).

Strategy
--------
- All rolling window operations (means, stds, sums) run as CuPy ufuncs on GPU
- Data is transferred to GPU once at the start, all features computed on-device,
  then transferred back to CPU as a single NumPy array at the end
- Minimises PCIe transfer overhead (1 host→device, 1 device→host per pipeline)
- Falls back to NumPy/pandas automatically if no GPU is available

GPU operations covered
----------------------
  Technical:      SMA, EMA, MACD, Bollinger Bands, ATR, RSI, ROC, OBV, VWAP,
                  Stochastic, Williams %R, CCI, MFI, realised vol, candle features
  Microstructure: OFI, VPIN, Amihud, Kyle's lambda, bid-ask spread proxy,
                  realised variance, bipower variation, intraday seasonality
  StatArb:        Z-score, CUSUM, Kalman Filter, rolling z-score of returns

GPU-parallel (Numba CUDA) — still on CPU via Numba parallel:
  Hurst, OU half-life, ADF, autocorrelation, Roll spread
  (These are inherently serial per-window; GPU parallelism is across windows
   which Numba prange already handles on CPU cores)

Usage
-----
    from aiquant.features.gpu_features import build_features_gpu
    df_feat = build_features_gpu(df_raw)   # auto-detects GPU
"""

import numpy as np
import pandas as pd
import time
import platform
import logging
import warnings
warnings.filterwarnings('ignore', message='.*highly fragmented.*', category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)


# ── Colour helpers ────────────────────────────────────────────────────────────
def _c(code, text):
    if platform.system() == 'Windows':
        return text
    return f'\033[{code}m{text}\033[0m'

_G  = lambda t: _c('32', t)
_C  = lambda t: _c('36', t)
_Y  = lambda t: _c('33', t)
_D  = lambda t: _c('2',  t)
_B  = lambda t: _c('1',  t)


# ── GPU detection ─────────────────────────────────────────────────────────────

def detect_gpu():
    """
    Detect available GPU acceleration.
    Returns ('cupy', xp) | ('torch', xp) | ('cpu', np)
    """
    # Try CuPy first (best for array operations)
    try:
        import cupy as cp
        cp.array([1.0])  # test allocation
        mem = cp.cuda.Device(0).mem_info
        total_gb = mem[1] / 1e9
        logger.info(f"CuPy GPU detected: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()} ({total_gb:.1f} GB)")
        return 'cupy', cp
    except Exception:
        pass

    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"PyTorch CUDA detected: {torch.cuda.get_device_name(0)}")
            return 'torch', None  # torch used separately for ML
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("PyTorch MPS (Apple Silicon) detected")
            return 'mps', None
    except Exception:
        pass

    logger.info("No GPU detected — using CPU (NumPy + Numba parallel)")
    return 'cpu', np


# ── CuPy rolling helpers ──────────────────────────────────────────────────────

def _cp_rolling_mean(arr, window):
    """Rolling mean via CuPy cumsum trick — O(n) instead of O(n*w)."""
    import cupy as cp
    cs  = cp.cumsum(cp.concatenate([cp.zeros(1), arr]))
    out = cp.full(len(arr), cp.nan)
    out[window - 1:] = (cs[window:] - cs[:-window]) / window
    return out


def _cp_rolling_std(arr, window):
    """Rolling std via CuPy — Welford-style via mean of squares."""
    import cupy as cp
    n   = len(arr)
    out = cp.full(n, cp.nan)
    # Use rolling mean of x and x^2
    mean_x  = _cp_rolling_mean(arr, window)
    mean_x2 = _cp_rolling_mean(arr ** 2, window)
    var = mean_x2 - mean_x ** 2
    var = cp.maximum(var, 0.0)   # numerical safety
    out[window - 1:] = cp.sqrt(var[window - 1:])
    return out


def _cp_rolling_sum(arr, window):
    """Rolling sum via CuPy cumsum trick."""
    import cupy as cp
    cs  = cp.cumsum(cp.concatenate([cp.zeros(1), arr]))
    out = cp.full(len(arr), cp.nan)
    out[window - 1:] = cs[window:] - cs[:-window]
    return out


def _cp_ema(arr, span):
    """Exponential moving average via CuPy — sequential but on GPU memory."""
    import cupy as cp
    alpha = 2.0 / (span + 1)
    out   = cp.empty_like(arr)
    out[0] = arr[0]
    # CuPy does not have a native EMA; use a Python loop over GPU scalars
    # For large arrays this is still faster than pandas due to GPU memory locality
    # For very large arrays, consider cupy.ElementwiseKernel
    arr_cpu = cp.asnumpy(arr)
    out_cpu = np.empty_like(arr_cpu)
    out_cpu[0] = arr_cpu[0]
    for i in range(1, len(arr_cpu)):
        out_cpu[i] = alpha * arr_cpu[i] + (1 - alpha) * out_cpu[i - 1]
    return cp.asarray(out_cpu)


def _cp_rsi(close, window=14):
    """RSI via CuPy."""
    import cupy as cp
    delta  = cp.diff(close)
    gain   = cp.maximum(delta, 0.0)
    loss   = cp.maximum(-delta, 0.0)
    avg_g  = _cp_rolling_mean(gain, window)
    avg_l  = _cp_rolling_mean(loss, window)
    rs     = avg_g / cp.where(avg_l == 0, 1e-10, avg_l)
    rsi    = cp.full(len(close), cp.nan)
    rsi[1:] = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ── GPU feature builder ───────────────────────────────────────────────────────

def _build_technical_gpu(df: pd.DataFrame, cp) -> pd.DataFrame:
    """Build technical features on GPU using CuPy."""
    n = len(df)

    # Transfer OHLCV to GPU
    o = cp.asarray(df['open'].to_numpy(dtype=np.float64))
    h = cp.asarray(df['high'].to_numpy(dtype=np.float64))
    l = cp.asarray(df['low'].to_numpy(dtype=np.float64))
    c = cp.asarray(df['close'].to_numpy(dtype=np.float64))
    v = cp.asarray(df['volume'].to_numpy(dtype=np.float64))

    feats = {}

    # ── Moving averages ───────────────────────────────────────────────────
    for p in [7, 14, 21, 50, 100, 200]:
        feats[f'sma_{p}'] = _cp_rolling_mean(c, p)
        feats[f'ema_{p}'] = _cp_ema(c, p)

    for p in [21, 50, 200]:
        sma = feats[f'sma_{p}']
        feats[f'close_vs_sma{p}'] = (c - sma) / cp.where(sma == 0, 1e-10, sma)

    # ── MACD ──────────────────────────────────────────────────────────────
    for fast, slow, sig in [(12, 26, 9), (5, 13, 4), (3, 10, 16)]:
        ema_f = _cp_ema(c, fast)
        ema_s = _cp_ema(c, slow)
        macd  = ema_f - ema_s
        signal_line = _cp_ema(macd, sig)
        tag = f"{fast}_{slow}"
        feats[f'macd_{tag}']        = macd
        feats[f'macd_signal_{tag}'] = signal_line
        feats[f'macd_diff_{tag}']   = macd - signal_line

    # ── RSI ───────────────────────────────────────────────────────────────
    for p in [7, 14, 21]:
        feats[f'rsi_{p}'] = _cp_rsi(c, p)

    # ── Returns and log returns ───────────────────────────────────────────
    log_ret = cp.full(n, cp.nan)
    log_ret[1:] = cp.log(c[1:] / c[:-1])
    ret = cp.full(n, cp.nan)
    ret[1:] = (c[1:] - c[:-1]) / c[:-1]
    feats['returns']     = ret
    feats['log_returns'] = log_ret

    # ── Realised volatility ───────────────────────────────────────────────
    for p in [20, 60, 120, 240]:
        feats[f'realvol_{p}'] = _cp_rolling_std(ret, p) * cp.sqrt(525_600.0)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    for std_mult in [1.5, 2.0, 2.5]:
        mu  = _cp_rolling_mean(c, 20)
        sig = _cp_rolling_std(c, 20)
        tag = str(std_mult).replace('.', '')
        upper = mu + std_mult * sig
        lower = mu - std_mult * sig
        width = upper - lower
        feats[f'bb_high_{tag}']  = upper
        feats[f'bb_low_{tag}']   = lower
        feats[f'bb_width_{tag}'] = width / cp.where(mu == 0, 1e-10, mu)
        feats[f'bb_pct_{tag}']   = (c - lower) / cp.where(width == 0, 1e-10, width)

    # ── ATR ───────────────────────────────────────────────────────────────
    prev_c = cp.concatenate([cp.array([c[0]]), c[:-1]])
    tr = cp.maximum(h - l, cp.maximum(cp.abs(h - prev_c), cp.abs(l - prev_c)))
    for p in [7, 14, 21]:
        feats[f'atr_{p}'] = _cp_rolling_mean(tr, p)

    # ── Stochastic ────────────────────────────────────────────────────────
    h14 = _cp_rolling_mean(h, 14)   # proxy (proper would be rolling max)
    l14 = _cp_rolling_mean(l, 14)
    denom_stoch = cp.where((h14 - l14) == 0, 1e-10, h14 - l14)
    feats['stoch_k'] = 100.0 * (c - l14) / denom_stoch
    feats['stoch_d'] = _cp_rolling_mean(feats['stoch_k'], 3)

    # ── Williams %R ───────────────────────────────────────────────────────
    feats['williams_r'] = -100.0 * (h14 - c) / denom_stoch

    # ── Rate of Change ────────────────────────────────────────────────────
    for p in [5, 10, 20]:
        roc = cp.full(n, cp.nan)
        roc[p:] = (c[p:] - c[:-p]) / cp.where(c[:-p] == 0, 1e-10, c[:-p]) * 100.0
        feats[f'roc_{p}'] = roc

    # ── OBV ───────────────────────────────────────────────────────────────
    sign_ret = cp.sign(ret)
    sign_ret[0] = 0.0
    feats['obv'] = cp.cumsum(sign_ret * v)

    # ── VWAP ─────────────────────────────────────────────────────────────
    typical = (h + l + c) / 3.0
    cum_tv  = cp.cumsum(typical * v)
    cum_v   = cp.cumsum(v)
    feats['vwap'] = cum_tv / cp.where(cum_v == 0, 1e-10, cum_v)
    feats['close_vs_vwap'] = (c - feats['vwap']) / cp.where(feats['vwap'] == 0, 1e-10, feats['vwap'])

    # ── MFI ───────────────────────────────────────────────────────────────
    tp_delta = cp.full(n, cp.nan)
    tp_delta[1:] = typical[1:] - typical[:-1]
    pos_flow = cp.where(tp_delta > 0, typical * v, 0.0)
    neg_flow = cp.where(tp_delta < 0, typical * v, 0.0)
    pos_sum  = _cp_rolling_sum(pos_flow, 14)
    neg_sum  = _cp_rolling_sum(neg_flow, 14)
    mfr      = pos_sum / cp.where(neg_sum == 0, 1e-10, neg_sum)
    feats['mfi'] = 100.0 - (100.0 / (1.0 + mfr))

    # ── Volume features ───────────────────────────────────────────────────
    vol_sma20 = _cp_rolling_mean(v, 20)
    vol_sma60 = _cp_rolling_mean(v, 60)
    vol_std60 = _cp_rolling_std(v, 60)
    feats['vol_sma20']  = vol_sma20
    feats['vol_sma60']  = vol_sma60
    feats['vol_ratio']  = v / cp.where(vol_sma20 == 0, 1e-10, vol_sma20)
    feats['vol_zscore'] = (v - vol_sma60) / cp.where(vol_std60 == 0, 1e-10, vol_std60)

    # ── Candle features ───────────────────────────────────────────────────
    feats['body']        = c - o
    feats['body_pct']    = (c - o) / cp.where(o == 0, 1e-10, o)
    feats['upper_wick']  = h - cp.maximum(o, c)
    feats['lower_wick']  = cp.minimum(o, c) - l
    feats['candle_range'] = h - l
    gap = cp.full(n, cp.nan)
    gap[1:] = o[1:] - c[:-1]
    feats['gap']         = gap
    feats['gap_pct']     = gap / cp.where(c == 0, 1e-10, c)
    feats['is_bullish']  = (c > o).astype(cp.int8)
    feats['is_bearish']  = (c < o).astype(cp.int8)
    # ── ADX (Average Directional Index) ──────────────────────────────────
    # True Range already computed above as `tr`
    prev_h = cp.concatenate([cp.array([h[0]]), h[:-1]])
    prev_l = cp.concatenate([cp.array([l[0]]), l[:-1]])
    dm_pos = cp.where((h - prev_h) > (prev_l - l), cp.maximum(h - prev_h, 0.0), 0.0)
    dm_neg = cp.where((prev_l - l) > (h - prev_h), cp.maximum(prev_l - l, 0.0), 0.0)
    for period in [14, 21]:
        atr_p  = _cp_rolling_mean(tr, period)
        di_pos = 100.0 * _cp_rolling_mean(dm_pos, period) / cp.where(atr_p == 0, 1e-10, atr_p)
        di_neg = 100.0 * _cp_rolling_mean(dm_neg, period) / cp.where(atr_p == 0, 1e-10, atr_p)
        di_sum = di_pos + di_neg
        dx     = 100.0 * cp.abs(di_pos - di_neg) / cp.where(di_sum == 0, 1e-10, di_sum)
        adx_p  = _cp_rolling_mean(dx, period)
        if period == 14:
            feats['adx']     = adx_p
            feats['adx_pos'] = di_pos
            feats['adx_neg'] = di_neg
        feats[f'adx_{period}'] = adx_p
    # ── Volatility regime ─────────────────────────────────────────────────
    rv20  = feats.get('realvol_20',  _cp_rolling_std(ret, 20)  * cp.sqrt(525_600.0))
    rv240 = feats.get('realvol_240', _cp_rolling_std(ret, 240) * cp.sqrt(525_600.0))
    feats['vol_regime'] = rv20 / cp.where(rv240 == 0, 1e-10, rv240)
    # ── Transfer all GPU arrays back to CPU at once ───────────────────────
    result = {k: cp.asnumpy(v_arr) for k, v_arr in feats.items()}

    # Assign all new columns at once via pd.concat to avoid DataFrame fragmentation
    new_cols = pd.DataFrame(result, index=df.index)
    df = pd.concat([df, new_cols], axis=1)

    return df


def _build_microstructure_gpu(df: pd.DataFrame, cp) -> pd.DataFrame:
    """Build microstructure features on GPU using CuPy."""
    n = len(df)

    c = cp.asarray(df['close'].to_numpy(dtype=np.float64))
    h = cp.asarray(df['high'].to_numpy(dtype=np.float64))
    l = cp.asarray(df['low'].to_numpy(dtype=np.float64))
    v = cp.asarray(df['volume'].to_numpy(dtype=np.float64))
    ret = cp.asarray(df['returns'].to_numpy(dtype=np.float64))
    log_ret = cp.asarray(df['log_returns'].to_numpy(dtype=np.float64))

    feats = {}

    # ── Order Flow Imbalance ──────────────────────────────────────────────
    tick_dir = cp.sign(cp.diff(c, prepend=c[0]))
    signed_v = tick_dir * v
    feats['tick_direction'] = tick_dir
    feats['signed_volume']  = signed_v

    for w in [5, 15, 30, 60]:
        ofi = _cp_rolling_sum(signed_v, w)
        vol_sum = _cp_rolling_sum(v, w)
        feats[f'ofi_{w}']      = ofi
        feats[f'ofi_norm_{w}'] = ofi / cp.where(vol_sum == 0, 1e-10, vol_sum)

    # ── Trade Imbalance (Lee-Ready proxy) ─────────────────────────────────
    hl_range = cp.where((h - l) == 0, 1e-10, h - l)
    buy_vol  = v * ((c - l) / hl_range)
    sell_vol = v * ((h - c) / hl_range)
    feats['buy_vol_proxy']  = buy_vol
    feats['sell_vol_proxy'] = sell_vol

    for w in [10, 30, 60]:
        total = _cp_rolling_sum(v, w)
        total = cp.where(total == 0, 1e-10, total)
        br = _cp_rolling_sum(buy_vol, w) / total
        sr = _cp_rolling_sum(sell_vol, w) / total
        feats[f'buy_ratio_{w}']        = br
        feats[f'sell_ratio_{w}']       = sr
        feats[f'trade_imbalance_{w}']  = br - sr

    # ── VPIN ─────────────────────────────────────────────────────────────
    buy_frac = cp.clip((c - l) / hl_range, 0.0, 1.0)
    buy_frac = cp.where(cp.isnan(buy_frac), 0.5, buy_frac)
    bv = buy_frac * v
    sv = (1.0 - buy_frac) * v
    abs_imb = cp.abs(bv - sv)
    window_vpin = 2500   # 50 buckets * 50 bars
    total_v = _cp_rolling_sum(v, window_vpin)
    feats['vpin'] = _cp_rolling_sum(abs_imb, window_vpin) / cp.where(total_v == 0, 1e-10, total_v)
    feats['buy_frac'] = buy_frac
    feats['buy_vol']  = bv
    feats['sell_vol_vpin'] = sv
    feats['abs_imbalance'] = abs_imb

    # ── Amihud Illiquidity ────────────────────────────────────────────────
    dollar_v   = c * v
    abs_ret    = cp.abs(ret)
    feats['dollar_volume'] = dollar_v
    feats['abs_return']    = abs_ret

    for w in [20, 60, 120]:
        ratio = abs_ret / cp.where(dollar_v == 0, 1e-10, dollar_v)
        feats[f'amihud_{w}'] = _cp_rolling_mean(ratio, w) * 1e6

    # ── Kyle's Lambda ─────────────────────────────────────────────────────
    price_change = cp.diff(c, prepend=c[0])
    for w in [20, 60]:
        # Rolling cov(price_change, signed_volume) / var(signed_volume)
        cov_num = _cp_rolling_mean(price_change * signed_v, w) - \
                  _cp_rolling_mean(price_change, w) * _cp_rolling_mean(signed_v, w)
        var_sv  = _cp_rolling_std(signed_v, w) ** 2
        feats[f'kyle_lambda_{w}'] = cov_num / cp.where(var_sv == 0, 1e-10, var_sv)

    # ── Corwin-Schultz spread proxy ───────────────────────────────────────
    log_hl  = cp.log(h / cp.where(l == 0, 1e-10, l))
    beta    = _cp_rolling_sum(log_hl ** 2, 2)
    # Simplified proxy: spread ≈ 2*(sqrt(beta/2) - sqrt(beta/4))
    spread  = cp.maximum(0.0, 2.0 * (cp.sqrt(cp.maximum(0.0, beta / 2.0)) -
                                      cp.sqrt(cp.maximum(0.0, beta / 4.0))))
    feats['cs_spread'] = spread

    # ── Realised variance and bipower variation ───────────────────────────
    lr2 = log_ret ** 2
    abs_lr = cp.abs(log_ret)
    for w in [20, 60, 120]:
        feats[f'rv_{w}']   = _cp_rolling_sum(lr2, w)
        bpv_prod = abs_lr * cp.concatenate([cp.array([0.0]), abs_lr[:-1]])
        feats[f'bpv_{w}']  = _cp_rolling_sum(bpv_prod, w) * (np.pi / 2)
        feats[f'jump_{w}'] = cp.maximum(0.0, feats[f'rv_{w}'] - feats[f'bpv_{w}'])

    # ── Intraday seasonality ──────────────────────────────────────────────
    idx = df.index
    hour = cp.asarray(idx.hour.to_numpy(dtype=np.float64))
    minute = cp.asarray(idx.minute.to_numpy(dtype=np.float64))
    dow  = cp.asarray(idx.dayofweek.to_numpy(dtype=np.float64))
    feats['hour']        = hour
    feats['minute']      = minute
    feats['day_of_week'] = dow
    feats['is_weekend']  = (dow >= 5).astype(cp.int8)
    feats['hour_sin']    = cp.sin(2 * np.pi * hour / 24.0)
    feats['hour_cos']    = cp.cos(2 * np.pi * hour / 24.0)
    feats['dow_sin']     = cp.sin(2 * np.pi * dow / 7.0)
    feats['dow_cos']     = cp.cos(2 * np.pi * dow / 7.0)
    feats['minute_sin']  = cp.sin(2 * np.pi * minute / 60.0)
    feats['minute_cos']  = cp.cos(2 * np.pi * minute / 60.0)

    # ── Transfer back to CPU ──────────────────────────────────────────────
    result = {k: cp.asnumpy(v_arr) for k, v_arr in feats.items()}
    new_cols = pd.DataFrame(result, index=df.index)
    df = pd.concat([df, new_cols], axis=1)

    return df


def _build_statarb_gpu(df: pd.DataFrame, cp) -> pd.DataFrame:
    """Build StatArb features — GPU for z-score/CUSUM, CPU Numba for Hurst/ADF."""
    n = len(df)

    c   = cp.asarray(df['close'].to_numpy(dtype=np.float64))
    ret = cp.asarray(df['returns'].to_numpy(dtype=np.float64))

    feats = {}

    # ── Z-score features (GPU) ────────────────────────────────────────────
    for w in [20, 60, 120, 240]:
        mu    = _cp_rolling_mean(c, w)
        sigma = _cp_rolling_std(c, w)
        feats[f'zscore_price_{w}'] = (c - mu) / cp.where(sigma == 0, 1e-10, sigma)

        mu_r    = _cp_rolling_mean(ret, w)
        sigma_r = _cp_rolling_std(ret, w)
        feats[f'zscore_returns_{w}'] = (ret - mu_r) / cp.where(sigma_r == 0, 1e-10, sigma_r)

    # ── CUSUM structural break (GPU) ──────────────────────────────────────
    # Expanding mean and std
    cs_ret = cp.nancumsum(ret)
    count  = cp.arange(1, n + 1, dtype=cp.float64)
    mu_exp = cs_ret / count
    # Expanding std via E[X^2] - E[X]^2
    cs_ret2 = cp.nancumsum(ret ** 2)
    mu2_exp = cs_ret2 / count
    var_exp = cp.maximum(0.0, mu2_exp - mu_exp ** 2)
    std_exp = cp.sqrt(var_exp)
    std_exp = cp.where(std_exp == 0, 1e-10, std_exp)
    standardised = (ret - mu_exp) / std_exp
    feats['cusum_pos']    = cp.nancumsum(cp.maximum(0.0, standardised))
    feats['cusum_neg']    = cp.nancumsum(cp.maximum(0.0, -standardised))
    feats['regime_break'] = ((feats['cusum_pos'] > 3.0) | (feats['cusum_neg'] > 3.0)).astype(cp.int8)

    # ── Transfer GPU features back — use pd.concat to avoid fragmentation ──
    result = {k: cp.asnumpy(v_arr) for k, v_arr in feats.items()}
    new_cols = pd.DataFrame(result, index=df.index)
    df = pd.concat([df, new_cols], axis=1)

    # ── CPU Numba parallel: Hurst, half-life, ADF, Kalman, Roll spread ────
    from ..utils.fast_math import (
        rolling_hurst_nb, rolling_half_life_nb,
        rolling_adf_fast_nb, kalman_filter_nb, rolling_roll_spread_nb,
        rolling_autocorr_nb,
    )
    close_np   = df['close'].to_numpy(dtype=np.float64)
    returns_np = df['returns'].to_numpy(dtype=np.float64)

    hurst_arr = rolling_hurst_nb(close_np, window=240, max_lag=20)
    bins   = np.array([0.0, 0.45, 0.55, 1.0])
    labels = np.array(['mean_reverting', 'random_walk', 'trending'])
    idx_r  = np.clip(np.digitize(hurst_arr, bins) - 1, 0, len(labels) - 1)

    adf_arr   = rolling_adf_fast_nb(close_np, window=240)
    km, kr, kz = kalman_filter_nb(close_np)
    hl_arr    = rolling_half_life_nb(close_np, window=240)
    rs_arr    = rolling_roll_spread_nb(close_np, window=20)

    # Collect all CPU-computed columns into a dict, then concat once
    cpu_cols = {
        'hurst':           hurst_arr,
        'regime':          labels[idx_r],
        'half_life':       hl_arr,
        'adf_pvalue':      adf_arr,
        'is_stationary':   (adf_arr < 0.05).astype(np.int8),
        'kalman_mean':     km,
        'kalman_residual': kr,
        'kalman_zscore':   kz,
        'roll_spread':     rs_arr,
    }
    for lag in [1, 2, 3, 5, 10]:
        cpu_cols[f'autocorr_{lag}'] = rolling_autocorr_nb(returns_np, window=60, lag=lag)

    # Single pd.concat — avoids all fragmentation warnings
    cpu_df = pd.DataFrame(cpu_cols, index=df.index)
    df = pd.concat([df, cpu_df], axis=1)

    return df


# ── Master GPU feature builder ────────────────────────────────────────────────

def build_features_gpu(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Build all features with maximum GPU utilisation.

    Pipeline
    --------
    1. Detect GPU (CuPy / PyTorch CUDA / CPU fallback)
    2. Transfer OHLCV to GPU once
    3. Compute all rolling/element-wise features on GPU (CuPy)
    4. Transfer results back to CPU in one batch
    5. Run Hurst/ADF/Kalman on CPU with Numba parallel (GPU not faster here)
    6. Drop NaN warmup rows

    Falls back to CPU (NumPy + Numba) if no GPU is available.
    """
    backend, xp = detect_gpu()
    n_bars = len(df)
    t_total = time.perf_counter()

    if verbose:
        gpu_label = {
            'cupy':  _C('CuPy CUDA'),
            'torch': _C('PyTorch CUDA'),
            'mps':   _C('PyTorch MPS'),
            'cpu':   _Y('CPU (NumPy + Numba parallel)'),
        }.get(backend, backend)
        print(f"\n  {_B('GPU Feature Engineering')}  ·  {n_bars:,} bars  ·  Backend: {gpu_label}")
        print(f"  {'─' * 66}")

    if backend == 'cupy':
        # ── Full GPU path ─────────────────────────────────────────────────
        import cupy as cp

        def _stage(label, fn, df_):
            cols_before = df_.shape[1]
            t0 = time.perf_counter()
            df_ = fn(df_, cp)
            elapsed = time.perf_counter() - t0
            added = df_.shape[1] - cols_before
            speed = n_bars / elapsed if elapsed > 0 else 0
            if verbose:
                print(f"    {_G('✓')} {label:<26} {_C(f'+{added} features'):>18}  "
                      f"{_D(f'{elapsed:.1f}s')}  {_D(f'{speed:,.0f} bars/s')}")
            return df_

        if verbose:
            print(f"    {_C('⚙')}  Transferring data to GPU ...", end='\r')
        t0 = time.perf_counter()
        df = _stage("Technical (GPU)",      _build_technical_gpu,      df)
        df = _stage("Microstructure (GPU)", _build_microstructure_gpu, df)
        df = _stage("StatArb (GPU+Numba)",  _build_statarb_gpu,        df)

    else:
        # ── CPU fallback — use existing verbose pipeline ───────────────────
        if verbose:
            print(f"    {_Y('!')}  No GPU — falling back to CPU (NumPy + Numba parallel)")
        from . import build_full_feature_set
        df = build_full_feature_set(df, verbose=verbose)
        return df   # already drops NaN inside

    # ── Drop NaN warmup rows ──────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    total_elapsed = time.perf_counter() - t_total

    if verbose:
        print(f"  {'─' * 66}")
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        speed  = n_bars / total_elapsed if total_elapsed > 0 else 0
        print(
            f"  {_G('✓')} Done  ·  {_B(str(df.shape[1]))} features  ·  "
            f"{_B(f'{len(df):,}')} usable bars  {_D(f'({n_dropped} warmup rows dropped)')}"
        )
        print(
            f"  {_D('Memory:')} {mem_mb:.1f} MB  ·  "
            f"{_D('Total time:')} {total_elapsed:.1f}s  ·  "
            f"{_D('Throughput:')} {speed:,.0f} bars/s\n"
        )

    return df
