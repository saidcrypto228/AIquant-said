"""
aiquant/features/microstructure.py
====================================
Market microstructure features for HFT and statistical arbitrage.

Memory-efficient: each stage builds new columns as a dict, then pd.concat once
at the end — avoids DataFrame fragmentation and PerformanceWarning spam.

Implements:
  - Order Flow Imbalance (OFI)
  - Trade Imbalance (buy vs sell volume proxy)
  - VPIN (Volume-Synchronized Probability of Informed Trading)
  - Amihud Illiquidity Ratio
  - Kyle's Lambda (price impact)
  - Roll's Spread Estimator
  - Bid-Ask Spread Proxy (from OHLCV)
  - Realized Variance and Bipower Variation
  - Autocorrelation of Returns (short-term mean reversion signal)
  - Intraday Seasonality (time-of-day effects)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def order_flow_imbalance(df: pd.DataFrame, windows: list = [5, 15, 30, 60]) -> pd.DataFrame:
    tick_dir    = np.sign(df['close'] - df['close'].shift(1))
    signed_vol  = tick_dir * df['volume']
    new_cols = {
        'tick_direction': tick_dir,
        'signed_volume':  signed_vol,
    }
    for w in windows:
        ofi = signed_vol.rolling(w).sum()
        new_cols[f'ofi_{w}']      = ofi
        new_cols[f'ofi_norm_{w}'] = ofi / df['volume'].rolling(w).sum().replace(0, np.nan)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def trade_imbalance_proxy(df: pd.DataFrame, windows: list = [10, 30, 60]) -> pd.DataFrame:
    hl_range     = (df['high'] - df['low']).replace(0, np.nan)
    buy_vol_prx  = df['volume'] * ((df['close'] - df['low'])  / hl_range)
    sell_vol_prx = df['volume'] * ((df['high']  - df['close']) / hl_range)
    new_cols = {
        'buy_vol_proxy':  buy_vol_prx,
        'sell_vol_proxy': sell_vol_prx,
    }
    for w in windows:
        total = df['volume'].rolling(w).sum().replace(0, np.nan)
        br = buy_vol_prx.rolling(w).sum() / total
        sr = sell_vol_prx.rolling(w).sum() / total
        new_cols[f'buy_ratio_{w}']       = br
        new_cols[f'sell_ratio_{w}']      = sr
        new_cols[f'trade_imbalance_{w}'] = br - sr
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def vpin(df: pd.DataFrame, bucket_size: int = 50, n_buckets: int = 50) -> pd.DataFrame:
    hl_range     = (df['high'] - df['low']).replace(0, np.nan)
    buy_frac     = ((df['close'] - df['low']) / hl_range).fillna(0.5).clip(0, 1)
    buy_vol      = buy_frac * df['volume']
    sell_vol     = (1 - buy_frac) * df['volume']
    abs_imb      = (buy_vol - sell_vol).abs()
    window       = bucket_size * n_buckets
    total_vol    = df['volume'].rolling(window).sum().replace(0, np.nan)
    new_cols = {
        'buy_frac':      buy_frac,
        'buy_vol':       buy_vol,
        'sell_vol':      sell_vol,
        'abs_imbalance': abs_imb,
        'vpin':          abs_imb.rolling(window).sum() / total_vol,
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def amihud_illiquidity(df: pd.DataFrame, windows: list = [20, 60, 120]) -> pd.DataFrame:
    dollar_vol = df['close'] * df['volume']
    if 'returns' not in df.columns:
        returns = df['close'].pct_change()
    else:
        returns = df['returns']
    abs_ret = returns.abs()
    new_cols = {
        'dollar_volume': dollar_vol,
        'abs_return':    abs_ret,
    }
    for w in windows:
        new_cols[f'amihud_{w}'] = (
            abs_ret / dollar_vol.replace(0, np.nan)
        ).rolling(w).mean() * 1e6
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def kyles_lambda(df: pd.DataFrame, windows: list = [20, 60]) -> pd.DataFrame:
    if 'signed_volume' in df.columns:
        signed_vol = df['signed_volume']
    else:
        signed_vol = np.sign(df['close'] - df['close'].shift(1)) * df['volume']
    price_change = df['close'].diff()
    new_cols = {}
    if 'signed_volume' not in df.columns:
        new_cols['signed_volume'] = signed_vol
    for w in windows:
        cov = price_change.rolling(w).cov(signed_vol)
        var = signed_vol.rolling(w).var().replace(0, np.nan)
        new_cols[f'kyle_lambda_{w}'] = cov / var
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def roll_spread(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    from ..utils.fast_math import rolling_roll_spread_nb
    close    = df['close'].to_numpy(dtype=np.float64)
    roll_arr = rolling_roll_spread_nb(close, window=window)
    return pd.concat([df, pd.DataFrame({'roll_spread': roll_arr}, index=df.index)], axis=1)


def bid_ask_spread_proxy(df: pd.DataFrame) -> pd.DataFrame:
    log_hl = np.log(df['high'] / df['low'].replace(0, np.nan))
    beta   = (log_hl ** 2).rolling(2).sum()
    h_max  = df['high'].rolling(2).max()
    l_min  = df['low'].rolling(2).min()
    gamma  = (np.log(h_max / l_min.replace(0, np.nan))) ** 2
    alpha  = (np.sqrt(2 * beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2)) \
             - np.sqrt(gamma / (3 - 2 * np.sqrt(2)))
    spread = (2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))).clip(lower=0)
    return pd.concat([df, pd.DataFrame({'cs_spread': spread}, index=df.index)], axis=1)


def realized_variance(df: pd.DataFrame, windows: list = [20, 60, 120]) -> pd.DataFrame:
    if 'log_returns' in df.columns:
        log_ret = df['log_returns']
    else:
        log_ret = np.log(df['close'] / df['close'].shift(1))
    new_cols = {}
    if 'log_returns' not in df.columns:
        new_cols['log_returns'] = log_ret
    abs_ret = log_ret.abs()
    for w in windows:
        rv  = (log_ret ** 2).rolling(w).sum()
        bpv = (abs_ret * abs_ret.shift(1)).rolling(w).sum() * (np.pi / 2)
        new_cols[f'rv_{w}']   = rv
        new_cols[f'bpv_{w}']  = bpv
        new_cols[f'jump_{w}'] = np.maximum(0, rv - bpv)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def return_autocorrelation(df: pd.DataFrame, lags: list = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    from ..utils.fast_math import rolling_autocorr_nb
    if 'returns' in df.columns:
        returns_arr = df['returns'].to_numpy(dtype=np.float64)
    else:
        returns_arr = df['close'].pct_change().to_numpy(dtype=np.float64)
    new_cols = {}
    for lag in lags:
        new_cols[f'autocorr_{lag}'] = rolling_autocorr_nb(returns_arr, window=60, lag=lag)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def intraday_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    idx = df.index
    hour   = idx.hour
    minute = idx.minute
    dow    = idx.dayofweek
    # hour_sin/cos and dow_sin/cos are already added by technical.py — skip them
    # to avoid duplicate columns that break the float32 cast in run_ml_backtest.
    new_cols = {
        'hour':        hour,
        'minute':      minute,
        'day_of_week': dow,
        'is_weekend':  (dow >= 5).astype(np.int8),
        'minute_sin':  np.sin(2 * np.pi * minute / 60),
        'minute_cos':  np.cos(2 * np.pi * minute / 60),
    }
    # Only add hour/dow cyclical cols if technical.py didn't already add them
    if 'hour_sin' not in df.columns:
        new_cols['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        new_cols['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    if 'dow_sin' not in df.columns:
        new_cols['dow_sin'] = np.sin(2 * np.pi * dow / 7)
        new_cols['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def generate_all_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: apply all microstructure feature groups."""
    logger.info("Generating microstructure features...")
    df = order_flow_imbalance(df)
    df = trade_imbalance_proxy(df)
    df = vpin(df)
    df = amihud_illiquidity(df)
    df = kyles_lambda(df)
    df = roll_spread(df)
    df = bid_ask_spread_proxy(df)
    df = realized_variance(df)
    df = return_autocorrelation(df)
    df = intraday_seasonality(df)
    logger.info(f"Microstructure features complete. Shape: {df.shape}")
    return df
