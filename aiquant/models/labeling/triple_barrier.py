from __future__ import annotations

import numpy as np
import pandas as pd


def volatility_barriers(
    df: pd.DataFrame,
    pt_mult: float = 2.0,
    sl_mult: float = 1.0,
    max_holding: int = 48,
    vol_window: int = 100,
) -> pd.Series:
    """
    Triple Barrier Method labels.

    Labels:
        1  -> profit taking reached first
        0  -> no barrier / timeout
       -1  -> stop loss reached first

    Uses ATR-like volatility scaling.
    Designed for higher timeframe entry labeling.
    """

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    returns = pd.Series(close).pct_change()

    volatility = (
        returns
        .rolling(vol_window)
        .std()
        .fillna(method="bfill")
        .values
    )

    labels = np.zeros(len(df), dtype=np.int8)

    for i in range(len(df) - max_holding):

        entry = close[i]

        vol = volatility[i]

        if vol <= 0:
            continue

        upper = entry * (1 + pt_mult * vol)
        lower = entry * (1 - sl_mult * vol)

        future_high = high[i + 1:i + 1 + max_holding]
        future_low = low[i + 1:i + 1 + max_holding]

        hit_tp = np.where(future_high >= upper)[0]
        hit_sl = np.where(future_low <= lower)[0]

        tp_time = hit_tp[0] if len(hit_tp) else np.inf
        sl_time = hit_sl[0] if len(hit_sl) else np.inf

        if tp_time < sl_time:
            labels[i] = 1

        elif sl_time < tp_time:
            labels[i] = -1

        else:
            labels[i] = 0

    return pd.Series(
        labels,
        index=df.index,
        name="triple_barrier_label"
    )


def label_distribution(labels: pd.Series) -> dict:
    """
    Quick diagnostic.
    """

    return {
        "long": int((labels == 1).sum()),
        "flat": int((labels == 0).sum()),
        "short": int((labels == -1).sum()),
        "total": int(len(labels)),
    }