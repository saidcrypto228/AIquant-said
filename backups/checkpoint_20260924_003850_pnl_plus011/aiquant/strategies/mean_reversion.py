"""
aiquant/strategies/mean_reversion.py
======================================
BACKUP STRATEGY 1: Multi-Signal Mean Reversion on 1m BTCUSD.

Uses Bollinger Bands + RSI + Volume confirmation + Volatility filter.
Designed for high-frequency 1m data with tight risk controls.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """
    Multi-signal mean reversion strategy combining:
      - Bollinger Band position (price overextension)
      - RSI (momentum exhaustion)
      - Volume confirmation (capitulation/euphoria volume)
      - Volatility spike filter (avoid during extreme moves)
      - Returns z-score (standardised deviation)
    """

    def __init__(
        self,
        bb_pct_long: float = 0.05,
        bb_pct_short: float = 0.95,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        vol_ratio_min: float = 1.2,
        zscore_long: float = -2.0,
        zscore_short: float = 2.0,
        vol_spike_max: float = 3.0,
        rsi_period: int = 14,
    ):
        self.bb_pct_long = bb_pct_long
        self.bb_pct_short = bb_pct_short
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.vol_ratio_min = vol_ratio_min
        self.zscore_long = zscore_long
        self.zscore_short = zscore_short
        self.vol_spike_max = vol_spike_max
        self.rsi_period = rsi_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate mean-reversion signals using fully vectorised NumPy operations.
        Extracts all columns to contiguous float64 arrays before comparison
        to avoid pandas indexing overhead on large 1m datasets.
        """
        if len(df) == 0:
            return pd.Series(np.zeros(0, dtype=np.int8), index=df.index, name='signal')

        bb_col     = 'bb_pct_20'
        rsi_col    = f'rsi_{self.rsi_period}'
        zscore_col = 'zscore_returns_60'

        for col in [bb_col, rsi_col, 'vol_ratio', zscore_col, 'vol_regime']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")

        # Extract to contiguous NumPy float64 arrays
        bb_pct     = df[bb_col].to_numpy(dtype=np.float64)
        rsi        = df[rsi_col].to_numpy(dtype=np.float64)
        zscore     = df[zscore_col].to_numpy(dtype=np.float64)
        vol_ratio  = df['vol_ratio'].to_numpy(dtype=np.float64)
        vol_regime = df['vol_regime'].to_numpy(dtype=np.float64)

        # Volatility spike filter — avoid extreme vol environments
        vol_ok = vol_regime < self.vol_spike_max

        # Long: price overextended below lower BB, RSI oversold, vol confirmation
        long_entry = (
            vol_ok
            & (bb_pct   < self.bb_pct_long)
            & (rsi      < self.rsi_oversold)
            & (vol_ratio > self.vol_ratio_min)
            & (zscore   < self.zscore_long)
        )

        # Short: price overextended above upper BB, RSI overbought, vol confirmation
        short_entry = (
            vol_ok
            & (bb_pct   > self.bb_pct_short)
            & (rsi      > self.rsi_overbought)
            & (vol_ratio > self.vol_ratio_min)
            & (zscore   > self.zscore_short)
        )

        # Build int8 signal array
        signals_arr = np.zeros(len(df), dtype=np.int8)
        signals_arr[long_entry]  =  1
        signals_arr[short_entry] = -1

        logger.info(
            f"[MeanReversion] Long={int(np.sum(long_entry))}, "
            f"Short={int(np.sum(short_entry))}, Total bars={len(df)}"
        )
        return pd.Series(signals_arr, index=df.index, name='signal')
