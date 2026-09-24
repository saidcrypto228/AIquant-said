"""
aiquant/strategies/order_flow_momentum.py
==========================================
Order Flow Momentum Strategy
-----------------------------
Uses Order Flow Imbalance (OFI) as the primary signal. OFI measures the
net buying/selling pressure from the order book. Sustained positive OFI
predicts short-term price appreciation; sustained negative OFI predicts
short-term price decline.

This is a core HFT/market-microstructure strategy used by prop trading firms.
Unlike price-based momentum, OFI leads price rather than lagging it.

Signal logic
------------
- Long  when: OFI_norm_30 > threshold AND OFI_norm_5 > 0 (short-term confirmation)
              AND price above VWAP (trend confirmation)
              AND Hurst > 0.5 (trending regime)
- Short when: OFI_norm_30 < -threshold AND OFI_norm_5 < 0
              AND price below VWAP
              AND Hurst > 0.5

Academic basis: Cont et al. (2014), Kolm et al. (2023), Lehalle & Neuman (2019)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class OrderFlowMomentumStrategy:
    """
    Order flow imbalance momentum strategy.

    Parameters
    ----------
    ofi_threshold  : float  OFI_norm_30 threshold to trigger signal (default 0.15)
    hurst_min      : float  Minimum Hurst exponent for trending regime (default 0.50)
    hold_bars      : int    Bars to hold position before re-evaluating (default 15)
    use_vwap_filter: bool   Require price to be on correct side of VWAP (default True)
    """

    def __init__(
        self,
        ofi_threshold:   float = 0.15,
        hurst_min:       float = 0.50,
        hold_bars:       int   = 15,
        use_vwap_filter: bool  = True,
    ):
        self.ofi_threshold   = ofi_threshold
        self.hurst_min       = hurst_min
        self.hold_bars       = hold_bars
        self.use_vwap_filter = use_vwap_filter

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(df) == 0:
            return pd.Series(np.zeros(0, dtype=np.int8), index=df.index, name='signal')

        required = ['ofi_norm_30', 'ofi_norm_5', 'hurst']
        for col in required:
            if col not in df.columns:
                logger.warning(f"OFMomentum: Missing column: {col}. Setting to 0.")
                return pd.Series(0, index=df.index, dtype=np.int8)

        n        = len(df)
        ofi_30   = df['ofi_norm_30'].to_numpy(dtype=np.float64)
        ofi_5    = df['ofi_norm_5'].to_numpy(dtype=np.float64)
        hurst    = df['hurst'].to_numpy(dtype=np.float64)

        # Trending regime
        trending = hurst > self.hurst_min

        # VWAP filter
        if self.use_vwap_filter and 'vwap' in df.columns and 'close' in df.columns:
            close = df['close'].to_numpy(dtype=np.float64)
            vwap  = df['vwap'].to_numpy(dtype=np.float64)
            above_vwap = close > vwap
            below_vwap = close < vwap
        else:
            above_vwap = np.ones(n, dtype=bool)
            below_vwap = np.ones(n, dtype=bool)

        # Smooth OFI with 3-bar EMA to reduce noise
        ofi_30_smooth = pd.Series(ofi_30).ewm(span=3).mean().to_numpy()

        long_entry  = (ofi_30_smooth >  self.ofi_threshold) & (ofi_5 > 0) & trending & above_vwap
        short_entry = (ofi_30_smooth < -self.ofi_threshold) & (ofi_5 < 0) & trending & below_vwap

        # Flip signal: exit when OFI reverses or hold_bars elapsed
        signal    = np.zeros(n, dtype=np.int8)
        pos       = 0
        entry_bar = 0

        for i in range(n):
            if pos == 0:
                if long_entry[i]:
                    pos = 1; entry_bar = i
                elif short_entry[i]:
                    pos = -1; entry_bar = i
            elif pos == 1:
                timed_out = (i - entry_bar) >= self.hold_bars
                reversed_ = ofi_30_smooth[i] < -self.ofi_threshold * 0.5
                if timed_out or reversed_ or short_entry[i]:
                    pos = 0
            elif pos == -1:
                timed_out = (i - entry_bar) >= self.hold_bars
                reversed_ = ofi_30_smooth[i] >  self.ofi_threshold * 0.5
                if timed_out or reversed_ or long_entry[i]:
                    pos = 0
            signal[i] = pos

        return pd.Series(signal, index=df.index, name='ofm_signal')
