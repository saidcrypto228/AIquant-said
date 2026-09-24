"""
aiquant/strategies/ensemble.py
================================
Strategy Ensemble: combines all strategy signals with regime-adaptive weighting.

Priority:
  1. KalmanStatArb   (primary — HFT mean reversion)
  2. MeanReversion   (backup 1 — BB + RSI)
  3. TrendFollowing  (backup 2 — EMA crossover)
  4. MLSignal        (backup 3 — ensemble ML)

Regime routing:
  - Hurst < 0.45  → StatArb + MeanReversion dominant
  - Hurst 0.45–0.55 → ML signal dominant
  - Hurst > 0.55  → TrendFollowing dominant
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

from .stat_arb import KalmanStatArbStrategy
from .mean_reversion import MeanReversionStrategy
from .trend_following import TrendFollowingStrategy

logger = logging.getLogger(__name__)


class StrategyEnsemble:
    """
    Regime-adaptive strategy ensemble.
    """

    def __init__(
        self,
        statarb_weight: float = 0.5,
        mr_weight: float = 0.25,
        trend_weight: float = 0.15,
        ml_weight: float = 0.10,
        min_agreement: int = 1,  # Minimum strategies agreeing for a signal
    ):
        self.statarb_weight = statarb_weight
        self.mr_weight = mr_weight
        self.trend_weight = trend_weight
        self.ml_weight = ml_weight
        self.min_agreement = min_agreement

        self.statarb = KalmanStatArbStrategy()
        self.mr = MeanReversionStrategy()
        self.trend = TrendFollowingStrategy()

    def generate_signals(
        self,
        df: pd.DataFrame,
        ml_signals: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate weighted ensemble signals.
        Returns a DataFrame with individual and combined signals.
        """
        if len(df) == 0:
            empty = pd.DataFrame(index=df.index)
            empty['final_signal'] = pd.Series(np.zeros(0, dtype=np.int8), index=df.index)
            return empty

        result = pd.DataFrame(index=df.index)

        # Generate individual signals (with fallback on missing features)
        try:
            result['statarb'] = self.statarb.generate_signals(df)
        except (ValueError, KeyError) as e:
            logger.warning(f"StatArb signal failed: {e}. Setting to 0.")
            result['statarb'] = 0

        try:
            result['mr'] = self.mr.generate_signals(df)
        except (ValueError, KeyError) as e:
            logger.warning(f"MeanReversion signal failed: {e}. Setting to 0.")
            result['mr'] = 0

        try:
            result['trend'] = self.trend.generate_signals(df)
        except (ValueError, KeyError) as e:
            logger.warning(f"TrendFollowing signal failed: {e}. Setting to 0.")
            result['trend'] = 0

        if ml_signals is not None:
            result['ml'] = ml_signals.reindex(df.index).fillna(0)
        else:
            result['ml'] = 0

        # --- Weighted score (NumPy vectorised) ----------------------------
        # Extract signal columns to float64 arrays for fast dot-product scoring
        sa  = result['statarb'].to_numpy(dtype=np.float64)
        mr  = result['mr'].to_numpy(dtype=np.float64)
        tr  = result['trend'].to_numpy(dtype=np.float64)
        ml  = result['ml'].to_numpy(dtype=np.float64)

        # Default weighted score
        score = (
            sa * self.statarb_weight
            + mr * self.mr_weight
            + tr * self.trend_weight
            + ml * self.ml_weight
        )  # shape: (n,) float64

        # --- Regime-adaptive routing (NumPy in-place assignment) -----------
        if 'hurst' in df.columns:
            hurst = df['hurst'].to_numpy(dtype=np.float64)
            hurst = np.where(np.isnan(hurst), 0.5, hurst)  # fill NaN with 0.5

            mr_regime    = hurst < 0.45   # mean-reverting
            trend_regime = hurst > 0.55   # trending

            # Boost StatArb + MR weights in mean-reverting regime
            score = np.where(
                mr_regime,
                sa * 0.60 + mr * 0.30 + ml * 0.10,
                score
            )
            # Boost Trend + ML weights in trending regime
            score = np.where(
                trend_regime,
                tr * 0.60 + ml * 0.25 + mr * 0.15,
                score
            )

        result['score'] = score

        # --- Final signal: threshold on weighted score (NumPy) -------------
        final_arr = np.zeros(len(df), dtype=np.int8)
        final_arr[score >=  0.25] =  1
        final_arr[score <= -0.25] = -1
        result['final_signal'] = final_arr

        logger.info(
            f"[Ensemble] Final signals — "
            f"Long: {int(np.sum(final_arr == 1))}, "
            f"Short: {int(np.sum(final_arr == -1))}, "
            f"Total bars: {len(df)}"
        )
        return result
