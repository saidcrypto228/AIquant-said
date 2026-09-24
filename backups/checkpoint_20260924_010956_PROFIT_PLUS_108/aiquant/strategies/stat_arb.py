"""
aiquant/strategies/stat_arb.py
================================
PRIMARY STRATEGY: High-Frequency Statistical Arbitrage on BTCUSD 1m.

Strategy Logic:
  - Uses Kalman Filter dynamic mean + z-score for entry/exit
  - Regime filter: only trade when Hurst < 0.5 (mean-reverting)
  - Order flow confirmation: OFI must oppose the price deviation
  - Volatility filter: avoid entries during vol spikes (VPIN > threshold)
  - Multi-signal confirmation: ADF stationarity + half-life < 30 bars

Entry Rules (Long):
  kalman_zscore < -entry_threshold
  AND hurst < 0.5 (mean-reverting regime)
  AND ofi_15 > 0 (buying pressure building)
  AND adf_pvalue < 0.1 (stationary)
  AND vpin < vpin_threshold (not toxic flow)

Entry Rules (Short):
  kalman_zscore > entry_threshold
  AND hurst < 0.5
  AND ofi_15 < 0 (selling pressure building)
  AND adf_pvalue < 0.1
  AND vpin < vpin_threshold

Exit Rules:
  kalman_zscore crosses zero (mean reversion complete)
  OR stop-loss hit
  OR max holding period exceeded
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KalmanStatArbStrategy:
    """
    Kalman Filter-based Statistical Arbitrage Strategy.
    Primary strategy for the AIQuant HFT framework.
    """

    def __init__(
        self,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.3,
        hurst_max: float = 0.5,
        vpin_max: float = 0.7,
        adf_pvalue_max: float = 0.10,
        ofi_window: int = 15,
        max_hold_bars: int = 60,  # 60 minutes max hold for 1m bars
    ):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.hurst_max = hurst_max
        self.vpin_max = vpin_max
        self.adf_pvalue_max = adf_pvalue_max
        self.ofi_window = ofi_window
        self.max_hold_bars = max_hold_bars

    def _check_required_columns(self, df: pd.DataFrame):
        required = ['kalman_zscore', 'hurst', f'ofi_norm_{self.ofi_window}', 'adf_pvalue', 'vpin']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Run feature engineering first.")

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate entry signals: 1=Long, -1=Short, 0=Flat.

        All comparisons are performed on raw NumPy arrays to avoid
        pandas element-wise overhead on large 1m datasets (100k+ bars).
        """
        if len(df) == 0:
            return pd.Series(np.zeros(0, dtype=np.int8), index=df.index, name='signal')
        self._check_required_columns(df)
        ofi_col = f'ofi_norm_{self.ofi_window}'

        # Extract to contiguous NumPy float64 arrays
        zscore  = df['kalman_zscore'].to_numpy(dtype=np.float64)
        hurst   = df['hurst'].to_numpy(dtype=np.float64)
        ofi     = df[ofi_col].to_numpy(dtype=np.float64)
        adf_p   = df['adf_pvalue'].to_numpy(dtype=np.float64)
        vpin    = df['vpin'].to_numpy(dtype=np.float64)

        # Boolean masks — all vectorised NumPy operations
        regime_ok   = hurst < self.hurst_max           # mean-reverting regime
        stationary  = adf_p < self.adf_pvalue_max      # ADF stationarity confirmed
        liquid      = vpin  < self.vpin_max            # not toxic order flow
        base_filter = regime_ok & stationary & liquid  # combined gate

        long_entry  = base_filter & (zscore < -self.entry_threshold) & (ofi > 0)
        short_entry = base_filter & (zscore >  self.entry_threshold) & (ofi < 0)

        # Build int8 signal array (memory-efficient for 100k+ bars)
        signals_arr = np.zeros(len(df), dtype=np.int8)
        signals_arr[long_entry]  =  1
        signals_arr[short_entry] = -1

        logger.info(
            f"[KalmanStatArb] Signals generated: "
            f"Long={int(np.sum(long_entry))}, Short={int(np.sum(short_entry))}, "
            f"Total bars={len(df)}"
        )
        return pd.Series(signals_arr, index=df.index, name='signal')

    def generate_exit_signals(self, df: pd.DataFrame, entry_signals: pd.Series) -> pd.Series:
        """
        Generate exit signals based on mean reversion completion.
        Uses np.abs for vectorised absolute value — faster than pd.Series.abs()
        on large arrays.
        """
        zscore    = df['kalman_zscore'].to_numpy(dtype=np.float64)
        exits_arr = np.zeros(len(df), dtype=np.int8)
        exits_arr[np.abs(zscore) < self.exit_threshold] = 1
        return pd.Series(exits_arr, index=df.index, name='exit')

    def signal_summary(self, df: pd.DataFrame, signals: pd.Series) -> dict:
        """Return a summary of signal statistics using NumPy aggregations."""
        sig_arr   = signals.to_numpy(dtype=np.int8)
        hurst_arr = df['hurst'].to_numpy(dtype=np.float64)
        vpin_arr  = df['vpin'].to_numpy(dtype=np.float64)
        adf_arr   = df['adf_pvalue'].to_numpy(dtype=np.float64)
        return {
            'total_long':      int(np.sum(sig_arr == 1)),
            'total_short':     int(np.sum(sig_arr == -1)),
            'signal_rate_pct': float(np.mean(sig_arr != 0) * 100),
            'avg_hurst':       float(np.nanmean(hurst_arr)),
            'avg_vpin':        float(np.nanmean(vpin_arr)),
            'pct_stationary':  float(np.mean(adf_arr < self.adf_pvalue_max) * 100),
        }
