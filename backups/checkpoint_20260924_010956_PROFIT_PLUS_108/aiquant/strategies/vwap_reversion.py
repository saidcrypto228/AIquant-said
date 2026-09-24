"""
aiquant/strategies/vwap_reversion.py
=====================================
VWAP Reversion Strategy
-----------------------
One of the most consistently profitable intraday strategies used by institutional
desks. The core insight: price tends to revert to VWAP (the volume-weighted
average price) throughout the trading day. Extreme deviations from VWAP are
mean-reverting, especially in liquid markets like BTC.

Signal logic
------------
- Long  when: price is > N std below VWAP AND RSI oversold AND vol_regime < 1.5
- Short when: price is > N std above VWAP AND RSI overbought AND vol_regime < 1.5
- Exit  when: price crosses back through VWAP

Academic basis: Berkowitz et al. (1988), Madhavan (2002), Kissell (2013)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class VWAPReversionStrategy:
    """
    VWAP-based mean reversion strategy.

    Parameters
    ----------
    vwap_std_entry : float
        Number of rolling std deviations from VWAP to trigger entry (default 1.5)
    vwap_std_exit  : float
        Number of std deviations to exit (default 0.3 — near VWAP)
    rsi_oversold   : float
        RSI threshold for long entry confirmation (default 35)
    rsi_overbought : float
        RSI threshold for short entry confirmation (default 65)
    vol_regime_max : float
        Maximum vol_regime ratio to trade (avoid high-vol regimes, default 2.0)
    lookback       : int
        Rolling window for VWAP std calculation (default 60 bars = 1 hour)
    """

    def __init__(
        self,
        vwap_std_entry: float = 1.5,
        vwap_std_exit:  float = 0.3,
        rsi_oversold:   float = 35.0,
        rsi_overbought: float = 65.0,
        vol_regime_max: float = 2.0,
        lookback:       int   = 60,
    ):
        self.vwap_std_entry  = vwap_std_entry
        self.vwap_std_exit   = vwap_std_exit
        self.rsi_oversold    = rsi_oversold
        self.rsi_overbought  = rsi_overbought
        self.vol_regime_max  = vol_regime_max
        self.lookback        = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(df) == 0:
            return pd.Series(np.zeros(0, dtype=np.int8), index=df.index, name='signal')

        required = ['close', 'vwap', 'rsi_14', 'vol_regime']
        for col in required:
            if col not in df.columns:
                logger.warning(f"VWAPReversion: Missing column: {col}. Setting to 0.")
                return pd.Series(0, index=df.index, dtype=np.int8)

        n     = len(df)
        close = df['close'].to_numpy(dtype=np.float64)
        vwap  = df['vwap'].to_numpy(dtype=np.float64)
        rsi   = df['rsi_14'].to_numpy(dtype=np.float64)
        vol_r = df['vol_regime'].to_numpy(dtype=np.float64)

        # Rolling std of (close - vwap) deviation
        dev   = close - vwap
        dev_s = pd.Series(dev)
        roll_std = dev_s.rolling(self.lookback, min_periods=20).std().to_numpy()
        roll_std = np.where(roll_std == 0, 1e-10, roll_std)

        z_score = dev / roll_std   # how many std from VWAP

        # Conditions
        low_vol     = vol_r < self.vol_regime_max
        oversold    = rsi   < self.rsi_oversold
        overbought  = rsi   > self.rsi_overbought

        long_entry  = (z_score < -self.vwap_std_entry) & oversold    & low_vol
        short_entry = (z_score >  self.vwap_std_entry) & overbought  & low_vol
        near_vwap   = np.abs(z_score) < self.vwap_std_exit

        # Build signal with state machine (hold position until exit)
        signal = np.zeros(n, dtype=np.int8)
        pos = 0
        for i in range(n):
            if pos == 0:
                if long_entry[i]:
                    pos = 1
                elif short_entry[i]:
                    pos = -1
            elif pos == 1:
                if near_vwap[i] or short_entry[i]:
                    pos = 0
            elif pos == -1:
                if near_vwap[i] or long_entry[i]:
                    pos = 0
            signal[i] = pos

        return pd.Series(signal, index=df.index, name='vwap_signal')
