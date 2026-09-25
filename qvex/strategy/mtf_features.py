from __future__ import annotations

import numpy as np
import pandas as pd


def atr(
    df: pd.DataFrame,
    period: int = 14,
) -> pd.Series:
    """Average True Range using completed OHLC candles only."""

    prev_close = df["close"].shift(1)

    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        period,
        min_periods=period,
    ).mean()


def swing_structure(
    df: pd.DataFrame,
    lookback: int = 3,
) -> pd.DataFrame:
    """
    Detect confirmed swing highs/lows.

    A swing is confirmed only after `lookback` candles
    exist on both sides.
    """

    if lookback < 1:
        raise ValueError("lookback must be >= 1")

    high = df["high"]
    low = df["low"]

    window = 2 * lookback + 1

    rolling_high = high.rolling(
        window,
        center=True,
        min_periods=window,
    ).max()

    rolling_low = low.rolling(
        window,
        center=True,
        min_periods=window,
    ).min()

    swing_high = high.where(high.eq(rolling_high))
    swing_low = low.where(low.eq(rolling_low))

    # Confirmation becomes available only after the
    # required candles to the right have closed.
    swing_high = swing_high.shift(lookback)
    swing_low = swing_low.shift(lookback)

    return pd.DataFrame(
        {
            "swing_high": swing_high,
            "swing_low": swing_low,
        },
        index=df.index,
    )


def add_4h_features(
    df: pd.DataFrame,
    atr_period: int = 14,
    swing_lookback: int = 3,
) -> pd.DataFrame:
    """Build causal 4H regime and structure features."""

    result = df.copy()

    result["atr"] = atr(
        result,
        period=atr_period,
    )

    structure = swing_structure(
        result,
        lookback=swing_lookback,
    )

    result = result.join(structure)

    result["last_swing_high"] = (
        result["swing_high"].ffill()
    )

    result["last_swing_low"] = (
        result["swing_low"].ffill()
    )

    result["structure_regime"] = np.select(
        [
            result["close"] > result["last_swing_high"],
            result["close"] < result["last_swing_low"],
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    result["atr_pct"] = (
        result["atr"] / result["close"]
    )

    return result


def add_1h_setup_features(
    df_1h: pd.DataFrame,
    df_4h_features: pd.DataFrame,
    atr_period: int = 14,
    swing_lookback: int = 3,
) -> pd.DataFrame:
    """
    Build causal 1H setup features using completed 4H information.

    4H features are aligned backward: for each 1H candle,
    only the latest completed 4H candle at or before that timestamp
    can be used.
    """

    result = df_1h.copy()

    # Native 1H volatility and structure.
    result["atr"] = atr(
        result,
        period=atr_period,
    )

    structure = swing_structure(
        result,
        lookback=swing_lookback,
    )

    result = result.join(structure)

    result["last_swing_high"] = (
        result["swing_high"].ffill()
    )

    result["last_swing_low"] = (
        result["swing_low"].ffill()
    )

    result["structure_regime"] = np.select(
        [
            result["close"] > result["last_swing_high"],
            result["close"] < result["last_swing_low"],
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    result["atr_pct"] = (
        result["atr"] / result["close"]
    )

    # Align only completed 4H information.
    htf = df_4h_features[
        [
            "atr",
            "last_swing_high",
            "last_swing_low",
            "structure_regime",
        ]
    ].copy()

    htf = htf.rename(
        columns={
            "atr": "htf_atr",
            "last_swing_high": "htf_last_swing_high",
            "last_swing_low": "htf_last_swing_low",
            "structure_regime": "htf_structure_regime",
        }
    )

    result = pd.merge_asof(
        result.sort_index(),
        htf.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # Distance from current 1H price to the latest confirmed
    # 4H structure levels, normalized by 1H ATR.
    result["dist_to_htf_high_atr"] = (
        result["htf_last_swing_high"] - result["close"]
    ) / result["atr"]

    result["dist_to_htf_low_atr"] = (
        result["close"] - result["htf_last_swing_low"]
    ) / result["atr"]

    return result


def add_15m_entry_features(
    df_15m: pd.DataFrame,
    df_1h_features: pd.DataFrame,
    atr_period: int = 14,
    swing_lookback: int = 3,
) -> pd.DataFrame:
    """
    Build causal 15M entry-confirmation features.

    Only the latest completed 1H information at or before
    each 15M candle timestamp is used.
    """

    result = df_15m.copy()

    # Native 15M volatility and structure.
    result["atr"] = atr(
        result,
        period=atr_period,
    )

    structure = swing_structure(
        result,
        lookback=swing_lookback,
    )

    result = result.join(structure)

    result["last_swing_high"] = (
        result["swing_high"].ffill()
    )

    result["last_swing_low"] = (
        result["swing_low"].ffill()
    )

    result["structure_regime"] = np.select(
        [
            result["close"] > result["last_swing_high"],
            result["close"] < result["last_swing_low"],
        ],
        [
            1,
            -1,
        ],
        default=0,
    )

    result["atr_pct"] = (
        result["atr"] / result["close"]
    )

    # Align only completed 1H information.
    htf = df_1h_features[
        [
            "atr",
            "last_swing_high",
            "last_swing_low",
            "structure_regime",
            "htf_structure_regime",
        ]
    ].copy()

    htf = htf.rename(
        columns={
            "atr": "htf_atr",
            "last_swing_high": "htf_last_swing_high",
            "last_swing_low": "htf_last_swing_low",
            "structure_regime": "htf_structure_regime_1h",
        }
    )

    result = pd.merge_asof(
        result.sort_index(),
        htf.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward",
    )

    # Distance to 1H structure, normalized by 15M ATR.
    result["dist_to_1h_high_atr"] = (
        result["htf_last_swing_high"] - result["close"]
    ) / result["atr"]

    result["dist_to_1h_low_atr"] = (
        result["close"] - result["htf_last_swing_low"]
    ) / result["atr"]

    # Whether 15M and 1H structure agree.
    result["structure_alignment"] = (
        result["structure_regime"]
        * result["htf_structure_regime_1h"]
    )

    return result
