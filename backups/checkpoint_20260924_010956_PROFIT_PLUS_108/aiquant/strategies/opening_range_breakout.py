"""
aiquant/strategies/opening_range_breakout.py
=============================================
Opening Range Breakout (ORB) Strategy
--------------------------------------
A classic institutional intraday strategy. The "opening range" is the high/low
established in the first N minutes of the session. A breakout above the range
high is a long signal; below the range low is a short signal.

For 24/7 crypto markets, we use a rolling 4-hour "session" window instead of
a traditional market open, since BTC has no fixed open. The 00:00 UTC reset
is the most commonly used anchor in crypto ORB strategies.

Signal logic
------------
- Define range: high/low of first `range_bars` bars of each UTC day
- Long  when: close breaks above range_high AND volume confirms
- Short when: close breaks below range_low  AND volume confirms
- Stop  at:   opposite side of range (range_low for longs, range_high for shorts)
- Target:     range_size * profit_ratio from entry

Academic basis: Toby Crabel (1990), Bhattacharya & Kumar (2006)
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class OpeningRangeBreakoutStrategy:
    """
    Opening Range Breakout for 24/7 crypto markets.

    Parameters
    ----------
    range_bars      : int   Number of bars to define the opening range (default 30 = 30 min)
    vol_confirm_mult: float Volume must be > vol_confirm_mult * avg_vol to confirm (default 1.2)
    profit_ratio    : float Target = range_size * profit_ratio (default 1.5)
    max_hold_bars   : int   Maximum bars to hold position (default 120 = 2 hours)
    """

    def __init__(
        self,
        range_bars:       int   = 30,
        vol_confirm_mult: float = 1.2,
        profit_ratio:     float = 1.5,
        max_hold_bars:    int   = 120,
    ):
        self.range_bars       = range_bars
        self.vol_confirm_mult = vol_confirm_mult
        self.profit_ratio     = profit_ratio
        self.max_hold_bars    = max_hold_bars

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        if len(df) == 0:
            return pd.Series(np.zeros(0, dtype=np.int8), index=df.index, name='signal')

        required = ['close', 'high', 'low', 'volume']
        for col in required:
            if col not in df.columns:
                logger.warning(f"ORB: Missing column: {col}. Setting to 0.")
                return pd.Series(0, index=df.index, dtype=np.int8)

        n      = len(df)
        close  = df['close'].to_numpy(dtype=np.float64)
        high   = df['high'].to_numpy(dtype=np.float64)
        low    = df['low'].to_numpy(dtype=np.float64)
        volume = df['volume'].to_numpy(dtype=np.float64)

        # Rolling average volume (60-bar)
        vol_avg = pd.Series(volume).rolling(60, min_periods=10).mean().to_numpy()
        vol_avg = np.where(vol_avg == 0, 1e-10, vol_avg)

        # Identify UTC day boundaries using index if available
        if hasattr(df.index, 'hour'):
            # Use hour == 0 and minute == 0 as session start
            is_session_start = (df.index.hour == 0) & (df.index.minute == 0)
            session_start_idx = np.where(is_session_start)[0]
        else:
            # Fallback: every 1440 bars = 1 day
            session_start_idx = np.arange(0, n, 1440)

        # Build opening range high/low for each session
        range_high = np.full(n, np.nan)
        range_low  = np.full(n, np.nan)

        for start in session_start_idx:
            end = min(start + self.range_bars, n)
            if end - start < 5:
                continue
            r_high = np.max(high[start:end])
            r_low  = np.min(low[start:end])
            # Apply this range to the rest of the session
            next_session = session_start_idx[session_start_idx > start]
            sess_end = int(next_session[0]) if len(next_session) > 0 else n
            range_high[end:sess_end] = r_high
            range_low[end:sess_end]  = r_low

        # Forward-fill NaN ranges
        range_high = pd.Series(range_high).ffill().to_numpy()
        range_low  = pd.Series(range_low).ffill().to_numpy()
        range_size = range_high - range_low

        # Volume confirmation
        vol_ok = volume > (vol_avg * self.vol_confirm_mult)

        # Breakout conditions
        long_break  = (close > range_high) & vol_ok & (range_size > 0)
        short_break = (close < range_low)  & vol_ok & (range_size > 0)

        # Build signal with hold and stop logic
        signal    = np.zeros(n, dtype=np.int8)
        pos       = 0
        entry_bar = 0
        entry_px  = 0.0
        stop_px   = 0.0
        target_px = 0.0

        for i in range(n):
            if pos == 0:
                if long_break[i] and not np.isnan(range_high[i]):
                    pos       = 1
                    entry_bar = i
                    entry_px  = close[i]
                    stop_px   = range_low[i]
                    target_px = entry_px + range_size[i] * self.profit_ratio
                elif short_break[i] and not np.isnan(range_low[i]):
                    pos       = -1
                    entry_bar = i
                    entry_px  = close[i]
                    stop_px   = range_high[i]
                    target_px = entry_px - range_size[i] * self.profit_ratio
            elif pos == 1:
                hit_stop   = close[i] < stop_px
                hit_target = close[i] > target_px
                timed_out  = (i - entry_bar) >= self.max_hold_bars
                if hit_stop or hit_target or timed_out:
                    pos = 0
            elif pos == -1:
                hit_stop   = close[i] > stop_px
                hit_target = close[i] < target_px
                timed_out  = (i - entry_bar) >= self.max_hold_bars
                if hit_stop or hit_target or timed_out:
                    pos = 0
            signal[i] = pos

        return pd.Series(signal, index=df.index, name='orb_signal')
