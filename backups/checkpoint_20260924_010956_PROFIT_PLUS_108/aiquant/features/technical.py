"""
aiquant/features/technical.py
==============================
Fast technical indicator feature engineering — NO ta library.
All indicators computed directly via NumPy/pandas: ~50x faster
than ta.add_all_ta_features() on large datasets.

Indicators computed
-------------------
Trend    : SMA(5,10,20,50,100,200), EMA(9,14,21,50), DEMA, TEMA
Momentum : RSI(7,14,21), MACD(12,26,9 + 5,13,4), Stochastic(14,3)
           Williams%R, ROC(5,10,20), CCI(20), MFI(14), TSI
Volatility: ATR(7,14,21), Bollinger Bands(20,2), Keltner Channel
            Historical Vol(20,60,120,240), BB Width, BB %B, vol_regime
Volume   : OBV, VWAP, CMF, MFI, vol_ratio, vol_zscore
Trend str: ADX(14), DI+, DI-
Candle   : body, wicks, gap, is_bullish, consec_bull/bear
Seasonality: hour_sin/cos, dow_sin/cos
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# ── Low-level NumPy helpers ───────────────────────────────────────────────────

def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(com=period - 1, adjust=False).mean().to_numpy()
    avg_l = pd.Series(loss).ewm(com=period - 1, adjust=False).mean().to_numpy()
    rs    = np.where(avg_l == 0, 100.0, avg_g / (avg_l + 1e-12))
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14) -> np.ndarray:
    prev = np.concatenate([[close[0]], close[:-1]])
    tr   = np.maximum(high - low,
           np.maximum(np.abs(high - prev), np.abs(low - prev)))
    return pd.Series(tr).ewm(com=period - 1, adjust=False).mean().to_numpy()


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
         period: int = 14):
    """Returns (adx, di_plus, di_minus) as numpy arrays."""
    prev_h = np.concatenate([[high[0]],  high[:-1]])
    prev_l = np.concatenate([[low[0]],   low[:-1]])
    prev_c = np.concatenate([[close[0]], close[:-1]])
    tr      = np.maximum(high - low,
              np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))
    dm_pos  = np.where((high - prev_h) > (prev_l - low),
                        np.maximum(high - prev_h, 0.0), 0.0)
    dm_neg  = np.where((prev_l - low) > (high - prev_h),
                        np.maximum(prev_l - low, 0.0), 0.0)
    atr14   = pd.Series(tr).ewm(com=period - 1, adjust=False).mean().to_numpy()
    dmp14   = pd.Series(dm_pos).ewm(com=period - 1, adjust=False).mean().to_numpy()
    dmm14   = pd.Series(dm_neg).ewm(com=period - 1, adjust=False).mean().to_numpy()
    di_pos  = np.where(atr14 > 0, 100 * dmp14 / atr14, 0.0)
    di_neg  = np.where(atr14 > 0, 100 * dmm14 / atr14, 0.0)
    dx      = np.where((di_pos + di_neg) > 0,
                        100 * np.abs(di_pos - di_neg) / (di_pos + di_neg + 1e-12), 0.0)
    adx_val = pd.Series(dx).ewm(com=period - 1, adjust=False).mean().to_numpy()
    return adx_val, di_pos, di_neg


# ── Main feature builder ──────────────────────────────────────────────────────

def generate_all_technical_features(df: pd.DataFrame,
                                    verbose: bool = False) -> pd.DataFrame:
    """
    Compute all technical features and append to df.
    Input df must have columns: open, high, low, close, volume.
    Returns df with new feature columns appended (operates in-place).
    """
    import time
    t0 = time.time()

    o = df['open'].to_numpy(dtype=np.float64)
    h = df['high'].to_numpy(dtype=np.float64)
    l = df['low'].to_numpy(dtype=np.float64)
    c = df['close'].to_numpy(dtype=np.float64)
    v = df['volume'].to_numpy(dtype=np.float64)
    n = len(c)

    # ── Trend: SMA ────────────────────────────────────────────────────────────
    for p in [5, 10, 20, 21, 50, 100, 200]:
        df[f'sma_{p}'] = pd.Series(c).rolling(p, min_periods=1).mean().to_numpy()

    # ── Trend: EMA ────────────────────────────────────────────────────────────
    ema9  = _ema(c, 9)
    ema12 = _ema(c, 12)
    ema14 = _ema(c, 14)
    ema21 = _ema(c, 21)
    ema26 = _ema(c, 26)
    ema50 = _ema(c, 50)
    for name, arr in [('ema_9', ema9), ('ema_14', ema14),
                       ('ema_21', ema21), ('ema_50', ema50)]:
        df[name] = arr

    # DEMA and TEMA
    ema20  = _ema(c, 20)
    ema20e = _ema(ema20, 20)
    df['dema_20'] = 2 * ema20 - ema20e

    e9e   = _ema(ema9, 9)
    e9ee  = _ema(e9e, 9)
    df['tema_9'] = 3 * ema9 - 3 * e9e + e9ee

    # Price vs MAs
    for p in [20, 50, 200]:
        sma = df[f'sma_{p}'].to_numpy()
        df[f'close_vs_sma{p}'] = (c - sma) / (sma + 1e-10)

    # ── Momentum: RSI ─────────────────────────────────────────────────────────
    for p in [7, 14, 21]:
        df[f'rsi_{p}'] = _rsi(c, p)

    # ── Momentum: MACD ────────────────────────────────────────────────────────
    for fast, slow, sig in [(12, 26, 9), (5, 13, 4)]:
        ef = _ema(c, fast)
        es = _ema(c, slow)
        ml = ef - es
        sl = _ema(ml, sig)
        tag = f"{fast}_{slow}"
        df[f'macd_{tag}']        = ml
        df[f'macd_signal_{tag}'] = sl
        df[f'macd_diff_{tag}']   = ml - sl

    # ── Momentum: Stochastic ──────────────────────────────────────────────────
    low14  = pd.Series(l).rolling(14, min_periods=1).min().to_numpy()
    high14 = pd.Series(h).rolling(14, min_periods=1).max().to_numpy()
    denom  = np.where(high14 - low14 > 0, high14 - low14, 1e-10)
    stoch_k = 100 * (c - low14) / denom
    df['stoch_k'] = stoch_k
    df['stoch_d'] = pd.Series(stoch_k).rolling(3, min_periods=1).mean().to_numpy()

    # ── Momentum: Williams %R ─────────────────────────────────────────────────
    df['williams_r'] = -100 * (high14 - c) / denom

    # ── Momentum: ROC ─────────────────────────────────────────────────────────
    for p in [5, 10, 20]:
        prev = np.concatenate([np.full(p, c[0]), c[:-p]])
        df[f'roc_{p}'] = np.where(prev > 0, (c - prev) / prev * 100, 0.0)

    # ── Momentum: CCI ─────────────────────────────────────────────────────────
    tp     = (h + l + c) / 3.0
    tp_s   = pd.Series(tp)
    tp_ma  = tp_s.rolling(20, min_periods=1).mean().to_numpy()
    tp_mad = tp_s.rolling(20, min_periods=1).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True).to_numpy()
    df['cci'] = np.where(tp_mad > 0, (tp - tp_ma) / (0.015 * tp_mad + 1e-12), 0.0)

    # ── Momentum: MFI ─────────────────────────────────────────────────────────
    raw_mf = tp * v
    prev_tp = np.concatenate([[tp[0]], tp[:-1]])
    pos_mf  = np.where(tp > prev_tp, raw_mf, 0.0)
    neg_mf  = np.where(tp < prev_tp, raw_mf, 0.0)
    pos14   = pd.Series(pos_mf).rolling(14, min_periods=1).sum().to_numpy()
    neg14   = pd.Series(neg_mf).rolling(14, min_periods=1).sum().to_numpy()
    df['mfi'] = np.where(neg14 > 0, 100 - 100 / (1 + pos14 / (neg14 + 1e-12)), 100.0)

    # ── Momentum: TSI (True Strength Index) ───────────────────────────────────
    pc   = np.diff(c, prepend=c[0])
    tsi_num = _ema(_ema(pc, 25), 13)
    tsi_den = _ema(_ema(np.abs(pc), 25), 13)
    df['tsi'] = np.where(tsi_den > 0, 100 * tsi_num / tsi_den, 0.0)

    # ── Volatility: ATR ───────────────────────────────────────────────────────
    for p in [7, 14, 21]:
        df[f'atr_{p}'] = _atr(h, l, c, p)
    atr14 = df['atr_14'].to_numpy()

    # ── Volatility: Bollinger Bands ───────────────────────────────────────────
    bb_mid = pd.Series(c).rolling(20, min_periods=1).mean().to_numpy()
    bb_std = pd.Series(c).rolling(20, min_periods=1).std(ddof=0).to_numpy()
    bb_std = np.where(bb_std == 0, 1e-10, bb_std)
    for mult, tag in [(1.5, '15'), (2.0, '20'), (2.5, '25')]:
        df[f'bb_high_{tag}']  = bb_mid + mult * bb_std
        df[f'bb_low_{tag}']   = bb_mid - mult * bb_std
        df[f'bb_width_{tag}'] = (2 * mult * bb_std) / (bb_mid + 1e-10)
        df[f'bb_pct_{tag}']   = (c - (bb_mid - mult * bb_std)) / (2 * mult * bb_std + 1e-10)
    df['bb_mid'] = bb_mid

    # ── Volatility: Keltner Channel ───────────────────────────────────────────
    kc_mid = ema20
    df['kc_high'] = kc_mid + 2 * atr14
    df['kc_low']  = kc_mid - 2 * atr14
    df['kc_pct']  = (c - df['kc_low'].to_numpy()) / (df['kc_high'].to_numpy() - df['kc_low'].to_numpy() + 1e-10)

    # ── Volatility: Donchian Channel ──────────────────────────────────────────
    dc_high = pd.Series(h).rolling(20, min_periods=1).max().to_numpy()
    dc_low  = pd.Series(l).rolling(20, min_periods=1).min().to_numpy()
    df['dc_high']  = dc_high
    df['dc_low']   = dc_low
    df['dc_width'] = (dc_high - dc_low) / (bb_mid + 1e-10)

    # ── Volatility: Historical vol ────────────────────────────────────────────
    log_ret = np.diff(np.log(np.where(c > 0, c, 1e-10)), prepend=0.0)
    df['returns']     = np.diff(c, prepend=c[0]) / (np.concatenate([[c[0]], c[:-1]]) + 1e-10)
    df['log_returns'] = log_ret
    for p in [20, 60, 120, 240]:
        df[f'realvol_{p}'] = pd.Series(log_ret).rolling(p, min_periods=1).std().to_numpy() * np.sqrt(525_600)
    rv20  = df['realvol_20'].to_numpy()
    rv240 = df['realvol_240'].to_numpy()
    df['vol_regime'] = rv20 / (rv240 + 1e-10)

    # ── Volume: OBV ───────────────────────────────────────────────────────────
    sign = np.sign(np.diff(c, prepend=c[0]))
    obv  = np.cumsum(sign * v)
    df['obv']     = obv
    df['obv_ema'] = _ema(obv, 20)

    # ── Volume: VWAP (rolling 20-bar) ─────────────────────────────────────────
    pv = tp * v
    df['vwap'] = pd.Series(pv).rolling(20, min_periods=1).sum().to_numpy() / \
                 (pd.Series(v).rolling(20, min_periods=1).sum().to_numpy() + 1e-10)
    df['close_vs_vwap'] = (c - df['vwap'].to_numpy()) / (df['vwap'].to_numpy() + 1e-10)

    # ── Volume: CMF (Chaikin Money Flow) ──────────────────────────────────────
    mfv = ((c - l) - (h - c)) / (h - l + 1e-10) * v
    df['cmf'] = pd.Series(mfv).rolling(20, min_periods=1).sum().to_numpy() / \
                (pd.Series(v).rolling(20, min_periods=1).sum().to_numpy() + 1e-10)

    # ── Volume: ratios and z-score ────────────────────────────────────────────
    vol_sma20 = pd.Series(v).rolling(20, min_periods=1).mean().to_numpy()
    vol_sma60 = pd.Series(v).rolling(60, min_periods=1).mean().to_numpy()
    vol_std60 = pd.Series(v).rolling(60, min_periods=1).std().to_numpy()
    df['vol_sma20']  = vol_sma20
    df['vol_sma60']  = vol_sma60
    df['vol_ratio']  = v / (vol_sma20 + 1e-10)
    df['vol_zscore'] = (v - vol_sma60) / (vol_std60 + 1e-10)

    # Force Index
    df['fi'] = np.diff(c, prepend=c[0]) * v

    # ── Trend strength: ADX ───────────────────────────────────────────────────
    adx_val, di_pos, di_neg = _adx(h, l, c, 14)
    df['adx']     = adx_val
    df['adx_14']  = adx_val
    df['adx_pos'] = di_pos
    df['adx_neg'] = di_neg

    # ── Candle features ───────────────────────────────────────────────────────
    body       = c - o
    candle_rng = h - l + 1e-10
    df['body']        = body
    df['body_pct']    = body / (o + 1e-10)
    df['upper_wick']  = (h - np.maximum(c, o)) / candle_rng
    df['lower_wick']  = (np.minimum(c, o) - l) / candle_rng
    df['candle_range']= h - l
    prev_c            = np.concatenate([[c[0]], c[:-1]])
    df['gap']         = o - prev_c
    df['gap_pct']     = df['gap'] / (prev_c + 1e-10)
    is_bull           = (c > o).astype(np.int8)
    is_bear           = (c < o).astype(np.int8)
    df['is_bullish']  = is_bull
    df['is_bearish']  = is_bear

    # Consecutive candle direction
    df['consec_bull'] = df['is_bullish'].groupby(
        (df['is_bullish'] != df['is_bullish'].shift()).cumsum()
    ).cumcount() + 1
    df['consec_bear'] = df['is_bearish'].groupby(
        (df['is_bearish'] != df['is_bearish'].shift()).cumsum()
    ).cumcount() + 1

    # ── Intraday seasonality ──────────────────────────────────────────────────
    if hasattr(df.index, 'hour'):
        hour   = df.index.hour.to_numpy()
        minute = df.index.minute.to_numpy()
        tod    = hour + minute / 60.0
        df['hour_sin'] = np.sin(2 * np.pi * tod / 24)
        df['hour_cos'] = np.cos(2 * np.pi * tod / 24)
        df['dow_sin']  = np.sin(2 * np.pi * df.index.dayofweek.to_numpy() / 7)
        df['dow_cos']  = np.cos(2 * np.pi * df.index.dayofweek.to_numpy() / 7)

    elapsed = time.time() - t0
    n_feat  = len(df.columns) - 5  # exclude ohlcv
    if verbose:
        logger.info(f"Technical: +{n_feat} features in {elapsed:.1f}s ({int(n/elapsed):,} bars/s)")
    return df


# ── Compatibility aliases ─────────────────────────────────────────────────────

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    return generate_all_technical_features(df)

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    return df

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    return df

def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    return df

def build_full_feature_set(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Legacy compatibility wrapper."""
    from aiquant.features import build_full_feature_set as _build
    return _build(df, verbose=verbose)
