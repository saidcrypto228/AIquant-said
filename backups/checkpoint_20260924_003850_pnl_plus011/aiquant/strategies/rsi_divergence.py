"""
aiquant/strategies/rsi_divergence.py
=====================================
RSI Divergence + Multi-Timeframe Momentum Strategy
----------------------------------------------------
Combines two high-probability setups:

1. RSI Divergence: Price makes a new low/high but RSI does not → reversal signal
2. Multi-Timeframe Momentum: 1m signal confirmed by 5m and 15m momentum

RSI divergence is one of the most reliable reversal signals in technical analysis,
especially on liquid assets like BTC where institutional traders actively watch it.

Signal logic
------------
Bullish divergence:
  - Price makes lower low over last N bars
  - RSI makes higher low over same period
  - RSI < 40 (oversold zone)
  - Long signal

Bearish divergence:
  - Price makes higher high over last N bars
  - RSI makes lower high over same period
  - RSI > 60 (overbought zone)
  - Short signal

Multi-timeframe confirmation:
  - 5m trend (5-bar EMA vs 15-bar EMA on 1m data)
  - 15m trend (15-bar EMA vs 45-bar EMA on 1m data)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class RSIDivergenceStrategy:
    """
    RSI divergence with multi-timeframe confirmation.

    Parameters
    ----------
    lookback       : int   Bars to look back for divergence detection (default 20)
    rsi_oversold   : float RSI level for bullish divergence (default 40)
    rsi_overbought : float RSI level for bearish divergence (default 60)
    hold_bars      : int   Bars to hold position (default 20)
    mtf_confirm    : bool  Require multi-timeframe confirmation (default True)
    """

    def __init__(
        self,
        lookback:       int   = 20,
        rsi_oversold:   float = 40.0,
        rsi_overbought: float = 60.0,
        hold_bars:      int   = 20,
        mtf_confirm:    bool  = True,
    ):
        self.lookback       = lookback
        self.rsi_oversold   = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.hold_bars      = hold_bars
        self.mtf_confirm    = mtf_confirm

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(df) == 0:
            return pd.Series(np.zeros(0, dtype=np.int8), index=df.index, name='signal')

        required = ['close', 'rsi_14']
        for col in required:
            if col not in df.columns:
                logger.warning(f"RSIDivergence: Missing column: {col}. Setting to 0.")
                return pd.Series(0, index=df.index, dtype=np.int8)

        n     = len(df)
        close = df['close'].to_numpy(dtype=np.float64)
        rsi   = df['rsi_14'].to_numpy(dtype=np.float64)

        # Multi-timeframe trend (using EMA proxies on 1m data)
        if self.mtf_confirm:
            c_s = pd.Series(close)
            ema5  = c_s.ewm(span=5,  adjust=False).mean().to_numpy()
            ema15 = c_s.ewm(span=15, adjust=False).mean().to_numpy()
            ema45 = c_s.ewm(span=45, adjust=False).mean().to_numpy()
            # 5m equivalent trend
            trend_5m_bull  = ema5  > ema15
            trend_5m_bear  = ema5  < ema15
            # 15m equivalent trend
            trend_15m_bull = ema15 > ema45
            trend_15m_bear = ema15 < ema45
        else:
            trend_5m_bull  = np.ones(n, dtype=bool)
            trend_5m_bear  = np.ones(n, dtype=bool)
            trend_15m_bull = np.ones(n, dtype=bool)
            trend_15m_bear = np.ones(n, dtype=bool)

        # Detect divergence using rolling window
        bull_div = np.zeros(n, dtype=bool)
        bear_div = np.zeros(n, dtype=bool)
        lb = self.lookback

        for i in range(lb, n):
            price_window = close[i - lb:i + 1]
            rsi_window   = rsi[i - lb:i + 1]

            # Bullish divergence: price lower low, RSI higher low
            price_min_idx = np.argmin(price_window)
            rsi_min_idx   = np.argmin(rsi_window)
            if (price_min_idx < lb and  # price low was earlier
                rsi_min_idx > price_min_idx and  # RSI low is more recent
                price_window[-1] < price_window[price_min_idx] * 1.002 and  # price near low
                rsi_window[-1] > rsi_window[rsi_min_idx] * 1.02 and  # RSI higher
                rsi[i] < self.rsi_oversold):
                bull_div[i] = True

            # Bearish divergence: price higher high, RSI lower high
            price_max_idx = np.argmax(price_window)
            rsi_max_idx   = np.argmax(rsi_window)
            if (price_max_idx < lb and
                rsi_max_idx > price_max_idx and
                price_window[-1] > price_window[price_max_idx] * 0.998 and
                rsi_window[-1] < rsi_window[rsi_max_idx] * 0.98 and
                rsi[i] > self.rsi_overbought):
                bear_div[i] = True

        # Apply MTF confirmation
        long_entry  = bull_div & trend_5m_bull & trend_15m_bull
        short_entry = bear_div & trend_5m_bear & trend_15m_bear

        # Build signal
        signal    = np.zeros(n, dtype=np.int8)
        pos       = 0
        entry_bar = 0

        for i in range(n):
            if pos == 0:
                if long_entry[i]:
                    pos = 1; entry_bar = i
                elif short_entry[i]:
                    pos = -1; entry_bar = i
            else:
                timed_out = (i - entry_bar) >= self.hold_bars
                flip      = (pos == 1 and short_entry[i]) or (pos == -1 and long_entry[i])
                if timed_out or flip:
                    pos = 0
            signal[i] = pos

        return pd.Series(signal, index=df.index, name='rsi_div_signal')
