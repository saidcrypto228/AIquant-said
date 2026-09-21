from __future__ import annotations

import pandas as pd

from ..strategy.mtf import resample_ohlcv
from ..strategy.mtf_features import (
    add_4h_features,
    add_1h_setup_features,
    add_15m_entry_features,
)


def build_mtf_features(
    df: pd.DataFrame,
    include_15m: bool = False,
) -> pd.DataFrame:
    """
    Build causal MTF features from the source timeframe.

    The returned DataFrame keeps the original source rows and adds
    only completed higher-timeframe information.

    Default experiment:
        4H + 1H

    Optional:
        4H + 1H + 15M
    """

    result = df.sort_index().copy()

    # ---------------------------------------------------------
    # 4H
    # ---------------------------------------------------------
    df_4h = resample_ohlcv(result, "4h")
    features_4h = add_4h_features(df_4h)

    # ---------------------------------------------------------
    # 1H
    # ---------------------------------------------------------
    df_1h = resample_ohlcv(result, "1h")
    features_1h = add_1h_setup_features(
        df_1h,
        features_4h,
    )

    # Keep only useful causal MTF columns.
    mtf_4h = features_4h[
        [
            "atr",
            "last_swing_high",
            "last_swing_low",
            "structure_regime",
            "atr_pct",
        ]
    ].rename(
        columns={
            "atr": "mtf_4h_atr",
            "last_swing_high": "mtf_4h_last_swing_high",
            "last_swing_low": "mtf_4h_last_swing_low",
            "structure_regime": "mtf_4h_structure_regime",
            "atr_pct": "mtf_4h_atr_pct",
        }
    )

    mtf_1h = features_1h[
        [
            "atr",
            "last_swing_high",
            "last_swing_low",
            "structure_regime",
            "atr_pct",
            "htf_atr",
            "htf_last_swing_high",
            "htf_last_swing_low",
            "htf_structure_regime",
            "dist_to_htf_high_atr",
            "dist_to_htf_low_atr",
        ]
    ].rename(
        columns={
            "atr": "mtf_1h_atr",
            "last_swing_high": "mtf_1h_last_swing_high",
            "last_swing_low": "mtf_1h_last_swing_low",
            "structure_regime": "mtf_1h_structure_regime",
            "atr_pct": "mtf_1h_atr_pct",
            "htf_atr": "mtf_1h_4h_atr",
            "htf_last_swing_high": "mtf_1h_4h_last_swing_high",
            "htf_last_swing_low": "mtf_1h_4h_last_swing_low",
            "htf_structure_regime": "mtf_1h_4h_structure_regime",
            "dist_to_htf_high_atr": "mtf_1h_dist_to_4h_high_atr",
            "dist_to_htf_low_atr": "mtf_1h_dist_to_4h_low_atr",
        }
    )

    # ---------------------------------------------------------
    # Align completed 4H + 1H features back to source rows.
    # ---------------------------------------------------------
    result = pd.merge_asof(
        result.sort_index(),
        mtf_4h.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    result = pd.merge_asof(
        result.sort_index(),
        mtf_1h.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # ---------------------------------------------------------
    # Optional 15M experiment.
    # ---------------------------------------------------------
    if include_15m:
        features_15m = add_15m_entry_features(
            resample_ohlcv(result, "15m"),
            features_1h,
        )

        mtf_15m = features_15m[
            [
                "atr",
                "last_swing_high",
                "last_swing_low",
                "structure_regime",
                "atr_pct",
                "htf_atr",
                "htf_last_swing_high",
                "htf_last_swing_low",
                "htf_structure_regime_1h",
                "dist_to_1h_high_atr",
                "dist_to_1h_low_atr",
                "structure_alignment",
            ]
        ].rename(
            columns={
                "atr": "mtf_15m_atr",
                "last_swing_high": "mtf_15m_last_swing_high",
                "last_swing_low": "mtf_15m_last_swing_low",
                "structure_regime": "mtf_15m_structure_regime",
                "atr_pct": "mtf_15m_atr_pct",
                "htf_atr": "mtf_15m_1h_atr",
                "htf_last_swing_high": "mtf_15m_1h_last_swing_high",
                "htf_last_swing_low": "mtf_15m_1h_last_swing_low",
                "htf_structure_regime_1h": "mtf_15m_1h_structure_regime",
                "dist_to_1h_high_atr": "mtf_15m_dist_to_1h_high_atr",
                "dist_to_1h_low_atr": "mtf_15m_dist_to_1h_low_atr",
                "structure_alignment": "mtf_15m_structure_alignment",
            }
        )

        result = pd.merge_asof(
            result.sort_index(),
            mtf_15m.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )

    return result