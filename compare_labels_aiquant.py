"""
compare_labels_aiquant.py

Compare AIQuant label distribution:

A) Baseline:
   THRESHOLD = 0.0008

B) Current:
   dyn_thresh = max(0.0015, ATR14 / close)

Same:
- forward bars = 15
- LONG / SHORT / FLAT rules
- same dataset
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================

DATA_PATH = Path(
    "data/raw/BTCUSDT_1m.parquet"
)

FORWARD_BARS = 15

BASELINE_THRESHOLD = 0.0008

ATR_MIN_THRESHOLD = 0.0015


# =========================
# ATR
# =========================

def compute_atr(high, low, close, period=14):

    prev_close = np.concatenate(
        ([close[0]], close[:-1])
    )

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )

    atr = (
        pd.Series(tr)
        .ewm(
            com=period - 1,
            adjust=False
        )
        .mean()
        .to_numpy()
    )

    return atr



# =========================
# LABELS
# =========================

def make_labels(close, threshold):

    n = len(close)

    fwd_ret = np.zeros(n)

    for i in range(n - FORWARD_BARS):
        fwd_ret[i] = (
            close[i + FORWARD_BARS]
            -
            close[i]
        ) / close[i]


    labels = np.zeros(
        n,
        dtype=np.int8
    )

    labels[
        fwd_ret > threshold
    ] = 1

    labels[
        fwd_ret < -threshold
    ] = -1


    valid = np.zeros(
        n,
        dtype=bool
    )

    valid[
        :n-FORWARD_BARS
    ] = True


    return labels, valid



# =========================
# STATS
# =========================

def stats(labels, valid):

    x = labels[valid]

    total = len(x)

    return {

        "LONG":
            int((x == 1).sum()),

        "SHORT":
            int((x == -1).sum()),

        "FLAT":
            int((x == 0).sum()),

        "TOTAL":
            total
    }



def print_stats(name, s):

    print("\n" + "="*50)
    print(name)
    print("="*50)

    for k in [
        "LONG",
        "SHORT",
        "FLAT"
    ]:

        pct = (
            s[k]
            /
            s["TOTAL"]
            *
            100
        )

        print(
            f"{k:<8}"
            f"{s[k]:>12,}"
            f"   {pct:6.2f}%"
        )

    print(
        f"{'TOTAL':<8}"
        f"{s['TOTAL']:>12,}"
    )



def compare(a,b):

    print("\n")
    print("="*65)
    print("CURRENT - BASELINE DIFFERENCE")
    print("="*65)

    for k in [
        "LONG",
        "SHORT",
        "FLAT"
    ]:

        abs_diff = (
            b[k]-a[k]
        )

        pct_a = (
            a[k]
            /
            a["TOTAL"]
            *
            100
        )

        pct_b = (
            b[k]
            /
            b["TOTAL"]
            *
            100
        )

        print(
            f"{k:<8}"
            f"{abs_diff:+12,}"
            f"   "
            f"{pct_b-pct_a:+7.2f}%"
        )



# =========================
# MAIN
# =========================

def main():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            DATA_PATH
        )


    if DATA_PATH.suffix == ".parquet":

        df = pd.read_parquet(
            DATA_PATH
        )

    else:

        df = pd.read_csv(
            DATA_PATH
        )


    required = {
        "close",
        "high",
        "low"
    }


    missing = (
        required
        -
        set(df.columns)
    )


    if missing:

        raise Exception(
            f"Missing columns: {missing}"
        )


    close = (
        df["close"]
        .to_numpy(
            dtype=np.float64
        )
    )


    print(
        f"Dataset: {DATA_PATH}"
    )

    print(
        f"Bars: {len(close):,}"
    )



    # ---------------------
    # baseline
    # ---------------------

    base_threshold = np.full(
        len(close),
        BASELINE_THRESHOLD
    )


    base_labels, mask = make_labels(
        close,
        base_threshold
    )


    base_stats = stats(
        base_labels,
        mask
    )



    # ---------------------
    # current AIQuant
    # ---------------------

    if "atr_14" in df.columns:

        atr = (
            df["atr_14"]
            .to_numpy(
                dtype=np.float64
            )
        )

        print(
            "ATR source: dataframe atr_14"
        )

    else:

        print(
            "ATR source: calculated"
        )

        atr = compute_atr(
            df["high"].to_numpy(),
            df["low"].to_numpy(),
            close
        )


    atr_pct = atr / np.maximum(
        close,
        1e-6
    )


    current_threshold = np.maximum(
        ATR_MIN_THRESHOLD,
        atr_pct
    )


    print("\nCurrent threshold statistics")

    print(
        f"Mean   {current_threshold.mean():.6f}"
    )

    print(
        f"Median {np.median(current_threshold):.6f}"
    )

    print(
        f"Min    {current_threshold.min():.6f}"
    )

    print(
        f"Max    {current_threshold.max():.6f}"
    )


    atr_labels, atr_mask = make_labels(
        close,
        current_threshold
    )


    atr_stats = stats(
        atr_labels,
        atr_mask
    )


    print_stats(
        "BASELINE THRESHOLD 0.0008",
        base_stats
    )


    print_stats(
        "CURRENT AIQUANT ATR LABELING",
        atr_stats
    )


    compare(
        base_stats,
        atr_stats
    )



if __name__ == "__main__":
    main()