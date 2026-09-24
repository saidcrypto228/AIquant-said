"""
aiquant/strategies/adaptive_ensemble.py
========================================
Adaptive Strategy Ensemble
---------------------------
Combines all 7 strategies with regime-aware dynamic weighting.

Strategy roster
---------------
1. KalmanStatArb       — primary mean-reversion (Hurst < 0.45)
2. VWAPReversion       — VWAP mean-reversion (any regime, low vol)
3. OpeningRangeBreakout— session breakout (trending regime)
4. OrderFlowMomentum   — OFI-driven momentum (Hurst > 0.50)
5. VolatilityBreakout  — BB squeeze breakout (low vol → expansion)
6. RSIDivergence       — reversal with MTF confirmation
7. MeanReversion       — RSI/BB classic mean reversion
8. TrendFollowing      — EMA crossover trend

Regime detection
----------------
- Mean-reverting (Hurst < 0.45): weight StatArb + VWAP + MeanReversion
- Trending      (Hurst > 0.55): weight ORB + OFM + TrendFollowing
- Volatile      (vol_regime > 2): reduce all weights, use VolBreakout
- Neutral       (0.45 ≤ Hurst ≤ 0.55): equal weight all

Signal combination
------------------
Weighted vote → threshold → final signal
Confidence = |weighted_sum| / total_weight
Only trade when confidence > min_confidence
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AdaptiveStrategyEnsemble:
    """
    Regime-aware adaptive ensemble of all strategies.

    Parameters
    ----------
    min_confidence : float  Minimum confidence to generate a signal (default 0.30)
    use_all        : bool   Use all 8 strategies (default True)
    """

    def __init__(self, min_confidence: float = 0.30, use_all: bool = True):
        self.min_confidence = min_confidence
        self.use_all        = use_all
        self._init_strategies()

    def _init_strategies(self):
        from aiquant.strategies.stat_arb           import KalmanStatArbStrategy
        from aiquant.strategies.mean_reversion     import MeanReversionStrategy
        from aiquant.strategies.trend_following    import TrendFollowingStrategy
        from aiquant.strategies.vwap_reversion     import VWAPReversionStrategy
        from aiquant.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
        from aiquant.strategies.order_flow_momentum    import OrderFlowMomentumStrategy
        from aiquant.strategies.volatility_breakout    import VolatilityBreakoutStrategy
        from aiquant.strategies.rsi_divergence         import RSIDivergenceStrategy

        self.strategies = {
            'stat_arb':   KalmanStatArbStrategy(),
            'vwap_rev':   VWAPReversionStrategy(),
            'orb':        OpeningRangeBreakoutStrategy(),
            'ofm':        OrderFlowMomentumStrategy(),
            'vol_break':  VolatilityBreakoutStrategy(),
            'rsi_div':    RSIDivergenceStrategy(),
            'mean_rev':   MeanReversionStrategy(),
            'trend':      TrendFollowingStrategy(),
        }

    # ── Regime weights ────────────────────────────────────────────────────────
    REGIME_WEIGHTS = {
        'mean_reverting': {
            'stat_arb': 0.30, 'vwap_rev': 0.25, 'mean_rev': 0.20,
            'rsi_div':  0.15, 'orb': 0.03, 'ofm': 0.03,
            'vol_break': 0.02, 'trend': 0.02,
        },
        'trending': {
            'orb':       0.30, 'ofm': 0.25, 'trend': 0.20,
            'vol_break': 0.10, 'rsi_div': 0.07, 'stat_arb': 0.04,
            'vwap_rev':  0.02, 'mean_rev': 0.02,
        },
        'volatile': {
            'vol_break': 0.35, 'orb': 0.20, 'ofm': 0.15,
            'rsi_div':   0.10, 'trend': 0.10, 'stat_arb': 0.05,
            'vwap_rev':  0.03, 'mean_rev': 0.02,
        },
        'neutral': {
            'stat_arb':  0.15, 'vwap_rev': 0.15, 'orb': 0.13,
            'ofm':       0.13, 'vol_break': 0.12, 'rsi_div': 0.12,
            'mean_rev':  0.10, 'trend': 0.10,
        },
    }

    def _detect_regime(self, row_hurst: float, row_vol_regime: float) -> str:
        if row_vol_regime > 2.5:
            return 'volatile'
        if row_hurst < 0.45:
            return 'mean_reverting'
        if row_hurst > 0.55:
            return 'trending'
        return 'neutral'

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined signals from all strategies.

        Returns
        -------
        pd.DataFrame with columns: final_signal, confidence, regime,
                                   and individual strategy signals.
        """
        n = len(df)

        # Get hurst and vol_regime for regime detection
        hurst     = df['hurst'].to_numpy(dtype=np.float64)     if 'hurst'      in df.columns else np.full(n, 0.5)
        vol_regime = df['vol_regime'].to_numpy(dtype=np.float64) if 'vol_regime' in df.columns else np.ones(n)

        # Generate each strategy's signal
        strat_signals = {}
        for name, strat in self.strategies.items():
            try:
                sig = strat.generate_signals(df)
                strat_signals[name] = sig.to_numpy(dtype=np.float64)
            except Exception as e:
                logger.warning(f"AdaptiveEnsemble: {name} failed: {e}. Setting to 0.")
                strat_signals[name] = np.zeros(n, dtype=np.float64)

        # Compute regime-weighted vote for each bar
        final_signal = np.zeros(n, dtype=np.int8)
        confidence   = np.zeros(n, dtype=np.float32)
        regime_arr   = np.empty(n, dtype=object)

        for i in range(n):
            regime = self._detect_regime(hurst[i], vol_regime[i])
            weights = self.REGIME_WEIGHTS[regime]
            total_w = sum(weights.values())

            weighted_sum = sum(
                weights[name] * strat_signals[name][i]
                for name in weights
            )
            conf = abs(weighted_sum) / total_w
            regime_arr[i] = regime
            confidence[i] = conf

            if conf >= self.min_confidence:
                final_signal[i] = int(np.sign(weighted_sum))
            else:
                final_signal[i] = 0

        # Build output DataFrame
        out = pd.DataFrame(index=df.index)
        out['final_signal'] = final_signal
        out['confidence']   = confidence
        out['regime']       = regime_arr
        for name, arr in strat_signals.items():
            out[f'sig_{name}'] = arr.astype(np.int8)

        return out
