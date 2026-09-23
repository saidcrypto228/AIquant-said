"""
grid_label_thresholds.py

Compare label distribution for different thresholds.
Does NOT modify AIQuant code.
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =============================
# CONFIG
# =============================

DATA_PATH = Path("data/raw/BTCUSDT_1m.parquet")

FORWARD_BARS = 15

THRESHOLDS = [
    0.0008,
    0.0010,
    0.0012,
    0.0015,
    0.0018,
]


ATR_PERIOD = 14
ATR_MULT = 1.0


# =============================
# ATR
# =============================

def compute_atr(high, low, close, period=14):

    prev = np.concatenate(
        [[close[0]], close[:-1]]
    )

    tr = np.maximum(
        high-low,
        np.maximum(
            abs(high-prev),
            abs(low-prev)
        )
    )

    return (
        pd.Series(tr)
        .ewm(
            com=period-1,
            adjust=False
        )
        .mean()
        .values
    )


# =============================
# LABELS
# =============================

def make_labels(close, threshold):

    n=len(close)

    fwd=np.zeros(n)

    for i in range(n-FORWARD_BARS):
        fwd[i]=(close[i+FORWARD_BARS]-close[i])/close[i]


    labels=np.zeros(n,dtype=np.int8)

    labels[fwd>threshold]=1
    labels[fwd<-threshold]=-1


    valid=np.zeros(n,dtype=bool)
    valid[:n-FORWARD_BARS]=True

    return labels[valid]


def stats(labels):

    n=len(labels)

    long=np.sum(labels==1)
    short=np.sum(labels==-1)
    flat=np.sum(labels==0)


    return {
        "LONG":long,
        "LONG%":long/n*100,

        "SHORT":short,
        "SHORT%":short/n*100,

        "FLAT":flat,
        "FLAT%":flat/n*100,

        "DIR%":(long+short)/n*100
    }



# =============================
# MAIN
# =============================


def main():

    df=pd.read_parquet(DATA_PATH)


    close=df["close"].values.astype(float)

    print()
    print("="*70)
    print("AIQuant Label Threshold Grid Test")
    print("="*70)

    print(
        f"Dataset: {DATA_PATH}"
    )

    print(
        f"Bars: {len(df):,}"
    )


    results=[]


    # fixed thresholds

    for t in THRESHOLDS:

        labels=make_labels(
            close,
            np.full(len(close),t)
        )

        s=stats(labels)

        s["THRESHOLD"]=t
        s["TYPE"]="FIXED"

        results.append(s)



    # ATR current logic from ensemble_pipeline.py

    if "atr_14" in df.columns:

        atr=df["atr_14"].values

    else:

        atr=compute_atr(
            df.high.values,
            df.low.values,
            close,
            ATR_PERIOD
        )


    atr_threshold=np.maximum(
        0.0015,
        (atr/close)*ATR_MULT
    )


    labels=make_labels(
        close,
        atr_threshold
    )


    s=stats(labels)

    s["THRESHOLD"]="ATR dynamic"
    s["TYPE"]="CURRENT AIQUANT"

    results.append(s)



    out=pd.DataFrame(results)


    cols=[
        "TYPE",
        "THRESHOLD",
        "LONG%",
        "SHORT%",
        "FLAT%",
        "DIR%"
    ]


    print()

    print(
        out[cols]
        .to_string(
            index=False,
            formatters={
                "LONG%":"{:5.2f}".format,
                "SHORT%":"{:5.2f}".format,
                "FLAT%":"{:5.2f}".format,
                "DIR%":"{:5.2f}".format,
            }
        )
    )


    print()

    print(
        "Interpretation:"
    )

    print(
        "- DIR% = percentage of bars producing directional labels"
    )

    print(
        "- Higher threshold => fewer trades, more selective"
    )

    print(
        "- Lower threshold => more training examples, more noise"
    )



if __name__=="__main__":
    main()