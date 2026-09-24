from __future__ import annotations

import pandas as pd


TIMEFRAMES = {
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


REQUIRED_COLUMNS = {
    "open",
    "high",
    "low",
    "close",
    "volume",
}


def validate_ohlcv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing OHLCV columns: {sorted(missing)}"
        )

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "DataFrame index must be a DatetimeIndex"
        )

    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DatetimeIndex must be sorted ascending"
        )


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:
    """
    Aggregate completed 1m OHLCV candles into a higher timeframe.

    The resulting candle is only available at its closing timestamp.
    No forward filling is performed.
    """

    validate_ohlcv(df)

    if timeframe not in TIMEFRAMES:
        raise ValueError(
            f"Unsupported timeframe: {timeframe}. "
            f"Supported: {list(TIMEFRAMES)}"
        )

    rule = TIMEFRAMES[timeframe]

    result = (
        df[list(REQUIRED_COLUMNS)]
        .resample(
            rule,
            label="right",
            closed="right",
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )

    # A higher-timeframe candle is usable only when its complete
    # time interval is present in the source data.
    source_last = df.index[-1]

    if timeframe == "15m":
        duration = pd.Timedelta(minutes=15)
    elif timeframe == "1h":
        duration = pd.Timedelta(hours=1)
    elif timeframe == "4h":
        duration = pd.Timedelta(hours=4)
    elif timeframe == "1d":
        duration = pd.Timedelta(days=1)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    result = result[result.index <= source_last.floor(duration)]

    return result


def build_mtf_ohlcv(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build all supported higher-timeframe OHLCV datasets.
    """

    validate_ohlcv(df)

    return {
        timeframe: resample_ohlcv(df, timeframe)
        for timeframe in TIMEFRAMES
    }
