"""
aiquant/strategies/volatility_breakout.py
==========================================
Volatility Breakout Strategy (Bollinger Band Squeeze)
------------------------------------------------------
Trades the expansion of volatility after a period of compression.
When Bollinger Bands are unusually narrow (squeeze), a breakout in either
direction tends to be sustained. This is the "Keltner Squeeze" concept
used by professional traders.

Signal logic
------------
- Detect squeeze: BB_width < percentile_20 of rolling BB_width (low vol)
- Long  when: squeeze AND close breaks above BB_upper AND ADX rising AND OFI > 0
- Short when: squeeze AND close breaks below BB_lower AND ADX rising AND OFI < 0
- Exit  when: close crosses back inside bands OR ATR-based stop hit

Academic basis: Bollinger (2001), Connors & Alvarez (2009), Kaufman (2013)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class VolatilityBreakoutStrategy:
    """
    Bollinger Band squeeze breakout strategy.

    Parameters
    ----------
    bb_col          : str   Bollinger Band width column (default 'bb_width_20')
    squeeze_pct     : float Percentile below which width is considered squeezed (default 25)
    adx_min         : float Minimum ADX to confirm trend strength (default 20)
    hold_bars       : int   Bars to hold after breakout (default 30)
    atr_stop_mult   : float ATR multiplier for stop loss (default 2.0)
    """

    def __init__(
        self,
        bb_col:        str   = 'bb_width_20',
        squeeze_pct:   float = 25.0,
        adx_min:       float = 20.0,
        hold_bars:     int   = 30,
        atr_stop_mult: float = 2.0,
    ):
        self.bb_col        = bb_col
        self.squeeze_pct   = squeeze_pct
        self.adx_min       = adx_min
        self.hold_bars     = hold_bars
        self.atr_stop_mult = atr_stop_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(df) == 0:
            return pd.Series(np.zeros(0, dtype=np.int8), index=df.index, name='signal')

        # Determine BB columns
        bb_upper_col = self.bb_col.replace('width', 'high').replace('bb_width_', 'bb_high_')
        bb_lower_col = self.bb_col.replace('width', 'low').replace('bb_width_', 'bb_low_')

        # Map bb_width_20 → bb_high_20, bb_low_20
        # Our feature names: bb_high_20, bb_low_20, bb_width_20
        tag = self.bb_col.replace('bb_width_', '')
        bb_upper_col = f'bb_high_{tag}'
        bb_lower_col = f'bb_low_{tag}'

        required = [self.bb_col, 'close', 'adx', 'atr_14']
        for col in required:
            if col not in df.columns:
                logger.warning(f"VolBreakout: Missing column: {col}. Setting to 0.")
                return pd.Series(0, index=df.index, dtype=np.int8)

        n        = len(df)
        close    = df['close'].to_numpy(dtype=np.float64)
        bb_width = df[self.bb_col].to_numpy(dtype=np.float64)
        adx      = df['adx'].to_numpy(dtype=np.float64)
        atr      = df['atr_14'].to_numpy(dtype=np.float64)

        bb_upper = df[bb_upper_col].to_numpy(dtype=np.float64) if bb_upper_col in df.columns else close + atr
        bb_lower = df[bb_lower_col].to_numpy(dtype=np.float64) if bb_lower_col in df.columns else close - atr

        # OFI for directional confirmation
        ofi = df['ofi_norm_30'].to_numpy(dtype=np.float64) if 'ofi_norm_30' in df.columns else np.zeros(n)

        # Rolling percentile of BB width (squeeze detection)
        bb_series = pd.Series(bb_width)
        squeeze_threshold = bb_series.rolling(240, min_periods=60).quantile(self.squeeze_pct / 100).to_numpy()
        in_squeeze = bb_width < squeeze_threshold

        # ADX rising (trend strength increasing)
        adx_series = pd.Series(adx)
        adx_rising = (adx_series.diff(3) > 0).to_numpy()

        # Breakout conditions
        long_break  = in_squeeze & (close > bb_upper) & adx_rising & (adx > self.adx_min) & (ofi >= 0)
        short_break = in_squeeze & (close < bb_lower) & adx_rising & (adx > self.adx_min) & (ofi <= 0)

        # Signal with ATR stop
        signal    = np.zeros(n, dtype=np.int8)
        pos       = 0
        entry_bar = 0
        stop_px   = 0.0

        for i in range(n):
            if pos == 0:
                if long_break[i]:
                    pos = 1; entry_bar = i
                    stop_px = close[i] - self.atr_stop_mult * atr[i]
                elif short_break[i]:
                    pos = -1; entry_bar = i
                    stop_px = close[i] + self.atr_stop_mult * atr[i]
            elif pos == 1:
                timed_out = (i - entry_bar) >= self.hold_bars
                stopped   = close[i] < stop_px
                if timed_out or stopped or short_break[i]:
                    pos = 0
            elif pos == -1:
                timed_out = (i - entry_bar) >= self.hold_bars
                stopped   = close[i] > stop_px
                if timed_out or stopped or long_break[i]:
                    pos = 0
            signal[i] = pos

        return pd.Series(signal, index=df.index, name='volbreak_signal')
