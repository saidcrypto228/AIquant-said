"""
aiquant/strategies/trend_following.py
=======================================
BACKUP STRATEGY 2: Multi-Timeframe Trend Following on 1m BTCUSD.

Uses EMA crossover + ADX trend strength + MACD confirmation
+ Volume trend filter. Designed to capture momentum regimes
when the Hurst exponent indicates trending (H > 0.55).

Rewritten to use pure pandas boolean operations to avoid all
numpy shape/dtype issues with Arrow-backed DataFrames in Colab.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TrendFollowingStrategy:
    """
    Multi-timeframe trend following strategy:
      - Fast/slow EMA crossover (primary signal)
      - ADX > threshold (trend strength confirmation)
      - MACD histogram direction (momentum confirmation)
      - Hurst > 0.55 (trending regime filter)
      - Volume above average (volume confirmation)
    """

    def __init__(
        self,
        fast_ema: int = 14,
        slow_ema: int = 50,
        adx_threshold: float = 25.0,
        hurst_min: float = 0.55,
        vol_ratio_min: float = 1.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
    ):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.adx_threshold = adx_threshold
        self.hurst_min = hurst_min
        self.vol_ratio_min = vol_ratio_min
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trend-following signals.
        Uses pure pandas operations to avoid numpy shape/dtype issues
        with Arrow-backed DataFrames in Colab/newer pandas versions.
        Returns pd.Series of int8: 1=Long, -1=Short, 0=Flat.
        """
        # Guard: empty DataFrame
        if len(df) == 0:
            return pd.Series(dtype=np.int8, name='signal')

        fast_col = f'ema_{self.fast_ema}'
        slow_col = f'ema_{self.slow_ema}'
        macd_col = f'macd_diff_{self.macd_fast}_{self.macd_slow}'

        for col in [fast_col, slow_col, 'adx', 'hurst', 'vol_ratio']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Work on a clean float64 copy — avoids all Arrow dtype surprises
        ema_fast  = df[fast_col].astype('float64').fillna(0.0)
        ema_slow  = df[slow_col].astype('float64').fillna(0.0)
        adx       = df['adx'].astype('float64').fillna(0.0)
        hurst     = df['hurst'].astype('float64').fillna(0.5)
        vol_ratio = df['vol_ratio'].astype('float64').fillna(1.0)

        # Regime and strength filters (pandas boolean Series)
        trending     = hurst     > self.hurst_min
        strong_trend = adx       > self.adx_threshold
        vol_ok       = vol_ratio > self.vol_ratio_min

        # EMA crossover state
        ema_bull = ema_fast > ema_slow
        ema_bear = ema_fast < ema_slow

        # MACD confirmation
        if macd_col in df.columns:
            macd = df[macd_col].astype('float64').fillna(0.0)
            macd_bull = macd > 0
            macd_bear = macd < 0
        else:
            macd_bull = pd.Series(True,  index=df.index)
            macd_bear = pd.Series(True,  index=df.index)

        long_state  = trending & strong_trend & vol_ok & ema_bull & macd_bull
        short_state = trending & strong_trend & vol_ok & ema_bear & macd_bear

        # Crossover detection: transition from False->True only
        long_cross  = long_state  & (~long_state.shift(1,  fill_value=False))
        short_cross = short_state & (~short_state.shift(1, fill_value=False))

        # Build signal Series
        signal = pd.Series(0, index=df.index, dtype=np.int8)
        signal[long_cross]  =  1
        signal[short_cross] = -1

        logger.info(
            f"[TrendFollowing] Long crossovers={int(long_cross.sum())}, "
            f"Short crossovers={int(short_cross.sum())}, Total bars={len(df)}"
        )
        return signal.rename('signal')
