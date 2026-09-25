#!/usr/bin/env python3
"""
Long/Short Feature Store & Master Gate Pipeline for Hyperliquid Universe.
Labels both Long breakouts and Short breakdowns under symmetric Triple Barrier.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/universe")
TARGET_COINS = ["BTC", "ETH", "SOL", "HYPE", "NEAR", "SUI", "XRP", "DOGE", "UNI"]

SL_MULT = 1.30
TP1_MULT = 1.20
TIMEOUT_BARS = 24

def compute_single_asset_features(df: pd.DataFrame, coin: str) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    atr_fast = tr.rolling(window=6, min_periods=6).mean()
    atr_slow = tr.rolling(window=48, min_periods=48).mean()

    feats = pd.DataFrame(index=df.index)
    feats["timestamp"] = df["timestamp"]
    feats["datetime"] = df["datetime"]
    feats["coin"] = coin
    feats["open"] = df["open"]
    feats["high"] = df["high"]
    feats["low"] = df["low"]
    feats["close"] = df["close"]
    feats["volume"] = df["volume"]
    feats["atr_14"] = atr_14

    feats["ret_1h_atr"] = (c - c.shift(1)) / (atr_14 + 1e-8)
    feats["ret_4h_atr"] = (c - c.shift(4)) / (atr_14 + 1e-8)
    feats["ret_12h_atr"] = (c - c.shift(12)) / (atr_14 + 1e-8)
    feats["ret_24h_atr"] = (c - c.shift(24)) / (atr_14 + 1e-8)
    feats["log_ret_24h"] = np.log(c / c.shift(24))

    vol_mean = v.rolling(window=72, min_periods=72).mean()
    vol_std = v.rolling(window=72, min_periods=72).std()
    feats["vol_zscore_72"] = ((v - vol_mean) / (vol_std + 1e-8)).clip(-3.0, 5.0)

    delta = c.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    loss = ((-delta.where(delta < 0, 0.0))).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    feats["rsi_norm"] = (rsi - 50.0) / 50.0

    sma_20 = c.rolling(window=20, min_periods=20).mean()
    std_20 = c.rolling(window=20, min_periods=20).std()
    upper_b = sma_20 + 2.0 * std_20
    lower_b = sma_20 - 2.0 * std_20
    b_range = (upper_b - lower_b).replace(0, np.nan)
    feats["bollinger_pct_b"] = ((c - lower_b) / b_range).clip(-0.5, 1.5)

    feats["vol_compression"] = (atr_fast / (atr_slow + 1e-8)).clip(0.2, 3.0)
    range_safe = (h - l).replace(0, np.nan)
    feats["bar_pressure"] = ((2 * c - h - l) / range_safe).fillna(0.0)

    ema_20 = c.ewm(span=20, adjust=False).mean()
    ema_50 = c.ewm(span=50, adjust=False).mean()
    feats["trend_spread_atr"] = (ema_20 - ema_50) / (atr_14 + 1e-8)

    # Сетапы лонга и шорта
    feats["long_setup_raw"] = ((c > ema_50) & ((rsi < 48.0) | (l <= ema_20))).astype(int)
    feats["short_setup_raw"] = ((c < ema_50) & ((rsi > 52.0) | (h >= ema_20))).astype(int)

    return feats

def main():
    print("=" * 65)
    print("  СБОРКА ДВУСТОРОННЕГО (LONG/SHORT) FEATURE STORE")
    print("=" * 65)

    btc_path = DATA_DIR / "BTC_1h.csv"
    btc_raw = pd.read_csv(btc_path)
    btc_feats = compute_single_asset_features(btc_raw, "BTC")

    btc_c = btc_feats["close"]
    btc_atr = btc_feats["atr_14"]
    btc_ret_1h = btc_c - btc_c.shift(1)
    btc_ret_4h_pct = (btc_c - btc_c.shift(4)) / btc_c.shift(4)

    flash_dump = (btc_ret_1h < -1.5 * btc_atr) | (btc_ret_4h_pct < -0.025)
    flash_pump = (btc_ret_1h > 1.5 * btc_atr) | (btc_ret_4h_pct > 0.025)

    btc_bull = (btc_feats["trend_spread_atr"] > 0) & (~flash_dump)
    btc_bear = (btc_feats["trend_spread_atr"] < 0) & (~flash_pump)

    master_gate_df = pd.DataFrame({
        "timestamp": btc_feats["timestamp"],
        "btc_log_ret_24h": btc_feats["log_ret_24h"],
        "master_gate_long": btc_bull.astype(int),
        "master_gate_short": btc_bear.astype(int)
    })

    all_coin_dfs = []

    for coin in TARGET_COINS:
        file_path = DATA_DIR / f"{coin}_1h.csv"
        if not file_path.exists():
            continue

        raw = pd.read_csv(file_path)
        coin_feats = compute_single_asset_features(raw, coin)
        merged = pd.merge(coin_feats, master_gate_df, on="timestamp", how="inner")

        merged["relative_strength_24h"] = merged["log_ret_24h"] - merged["btc_log_ret_24h"]

        c_arr = merged["close"].to_numpy()
        h_arr = merged["high"].to_numpy()
        l_arr = merged["low"].to_numpy()
        o_arr = merged["open"].to_numpy()
        atr_arr = merged["atr_14"].to_numpy()

        mg_long = merged["master_gate_long"].to_numpy()
        mg_short = merged["master_gate_short"].to_numpy()
        raw_l = merged["long_setup_raw"].to_numpy()
        raw_s = merged["short_setup_raw"].to_numpy()
        rs_arr = merged["relative_strength_24h"].to_numpy()

        n = len(merged)
        side = np.zeros(n, dtype=int)       # +1 = Long, -1 = Short, 0 = None
        labels = np.full(n, np.nan)

        for i in range(n - TIMEOUT_BARS - 1):
            cur_side = 0
            # Лонг: Master Gate Long + Сетап + Относительная сила
            if mg_long[i] == 1 and raw_l[i] == 1 and rs_arr[i] > 0:
                cur_side = 1
            # Шорт: Master Gate Short + Сетап + Относительная слабость
            elif mg_short[i] == 1 and raw_s[i] == 1 and rs_arr[i] < 0:
                cur_side = -1

            if cur_side == 0:
                continue

            side[i] = cur_side
            entry = o_arr[i + 1]
            cur_atr = atr_arr[i]

            if cur_side == 1:
                sl_p = entry - (cur_atr * SL_MULT)
                tp1_p = entry + (cur_atr * TP1_MULT)
                success = 0
                for k in range(1, TIMEOUT_BARS + 1):
                    idx = i + k
                    if l_arr[idx] <= sl_p:
                        success = 0
                        break
                    if h_arr[idx] >= tp1_p:
                        success = 1
                        break
                labels[i] = success

            elif cur_side == -1:
                sl_p = entry + (cur_atr * SL_MULT)
                tp1_p = entry - (cur_atr * TP1_MULT)
                success = 0
                for k in range(1, TIMEOUT_BARS + 1):
                    idx = i + k
                    if h_arr[idx] >= sl_p:
                        success = 0
                        break
                    if l_arr[idx] <= tp1_p:
                        success = 1
                        break
                labels[i] = success

        merged["side"] = side
        merged["meta_label"] = labels
        clean_coin = merged.dropna(subset=["atr_14", "vol_zscore_72"]).reset_index(drop=True)
        all_coin_dfs.append(clean_coin)

        n_l = (clean_coin["side"] == 1).sum()
        n_s = (clean_coin["side"] == -1).sum()
        print(f"[+] {coin:<5}: {len(clean_coin):,} баров | Лонгов: {n_l:<4} | Шортов: {n_s:<4}")

    universe_df = pd.concat(all_coin_dfs, ignore_index=True)
    universe_df = universe_df.sort_values(["timestamp", "coin"]).reset_index(drop=True)

    save_path = DATA_DIR / "universe_ls_pooled.parquet"
    universe_df.to_parquet(save_path, index=False)

    print("=" * 65)
    print(f"Всего строк в двустороннем пуле: {len(universe_df):,}")
    print(f"Всего сетапов (Long + Short)   : {(universe_df['side'] != 0).sum():,}")
    print(f"Файл сохранен в                : {save_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
