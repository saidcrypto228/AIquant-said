"""
aiquant/features/statarb.py
============================
Statistical arbitrage and regime detection features.

Memory-efficient: each stage builds new columns as a dict, then pd.concat once
at the end — avoids DataFrame fragmentation and PerformanceWarning spam.

Implements:
  - Z-score of price relative to rolling mean/std
  - Half-life of mean reversion (Ornstein-Uhlenbeck)
  - Hurst Exponent (trend vs mean-reversion regime)
  - ADF stationarity test (rolling, Numba-accelerated)
  - Kalman Filter dynamic spread
  - Structural break detection (CUSUM)
"""

import pandas as pd
import numpy as np
import logging

from ..utils.fast_math import (
    rolling_hurst_nb,
    rolling_half_life_nb,
    rolling_adf_fast_nb,
    kalman_filter_nb,
    rolling_roll_spread_nb,
)

logger = logging.getLogger(__name__)


def zscore_features(df: pd.DataFrame, windows: list = [20, 60, 120, 240]) -> pd.DataFrame:
    close    = df['close']
    new_cols = {}
    for w in windows:
        mu    = close.rolling(w).mean()
        sigma = close.rolling(w).std().replace(0, np.nan)
        new_cols[f'zscore_price_{w}'] = (close - mu) / sigma
        if 'returns' in df.columns:
            r     = df['returns']
            mu_r  = r.rolling(w).mean()
            sig_r = r.rolling(w).std().replace(0, np.nan)
            new_cols[f'zscore_returns_{w}'] = (r - mu_r) / sig_r
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    lags = range(2, max_lag)
    tau  = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]


def rolling_hurst(df: pd.DataFrame, window: int = 240, max_lag: int = 20) -> pd.DataFrame:
    close     = df['close'].to_numpy(dtype=np.float64)
    hurst_arr = rolling_hurst_nb(close, window=window, max_lag=max_lag)
    bins      = np.array([0.0, 0.45, 0.55, 1.0])
    labels    = np.array(['mean_reverting', 'random_walk', 'trending'])
    idx       = np.clip(np.digitize(hurst_arr, bins) - 1, 0, len(labels) - 1)
    new_cols  = {
        'hurst':  hurst_arr,
        'regime': labels[idx],
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def rolling_half_life(df: pd.DataFrame, window: int = 240) -> pd.DataFrame:
    close  = df['close'].to_numpy(dtype=np.float64)
    hl_arr = rolling_half_life_nb(close, window=window)
    return pd.concat([df, pd.DataFrame({'half_life': hl_arr}, index=df.index)], axis=1)


def adf_test_rolling(df: pd.DataFrame, window: int = 240) -> pd.DataFrame:
    close   = df['close'].to_numpy(dtype=np.float64)
    adf_arr = rolling_adf_fast_nb(close, window=window)
    new_cols = {
        'adf_pvalue':    adf_arr,
        'is_stationary': (adf_arr < 0.05).astype(np.int8),
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def kalman_filter_spread(df: pd.DataFrame, delta: float = 1e-4) -> pd.DataFrame:
    close = df['close'].to_numpy(dtype=np.float64)
    kalman_mean, kalman_residual, kalman_zscore = kalman_filter_nb(close, delta=delta)
    new_cols = {
        'kalman_mean':     kalman_mean,
        'kalman_residual': kalman_residual,
        'kalman_zscore':   kalman_zscore,
    }
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def cusum_structural_break(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    if 'returns' in df.columns:
        returns = df['returns']
    else:
        returns = df['close'].pct_change()
    mu           = returns.expanding().mean()
    sigma        = returns.expanding().std().replace(0, np.nan)
    standardised = (returns - mu) / sigma
    cusum_pos    = standardised.clip(lower=0).cumsum()
    cusum_neg    = (-standardised).clip(lower=0).cumsum()
    new_cols = {
        'cusum_pos':    cusum_pos,
        'cusum_neg':    cusum_neg,
        'regime_break': ((cusum_pos > threshold) | (cusum_neg > threshold)).astype(np.int8),
    }
    if 'returns' not in df.columns:
        new_cols['returns'] = returns
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def generate_all_statarb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: apply all statistical arbitrage feature groups."""
    logger.info("Generating statistical arbitrage features...")
    df = zscore_features(df)
    df = rolling_hurst(df)
    df = rolling_half_life(df)
    df = adf_test_rolling(df)
    df = kalman_filter_spread(df)
    df = cusum_structural_break(df)
    logger.info(f"StatArb features complete. Shape: {df.shape}")
    return df
