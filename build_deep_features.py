#!/usr/bin/env python3
"""
Deep Feature Store Pipeline (2023-2024).
Generates aligned indicators, BTC Master Gate, Relative Strength,
and Pullback Triple Barrier labels for SOL, NEAR, SUI, ETH.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/history_deep")
TARGET_COINS = ["BTC", "SOL", "NEAR", "SUI", "ETH"]

SL_MULT = 0.90
TP_MULT = 2.20
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

    ema_20 = c.ewm(span=20, adjust=False).mean()
    ema_50 = c.ewm(span=50, adjust=False).mean()

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
    feats["ema_20"] = ema_20
    feats["ema_50"] = ema_50

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
    feats["trend_spread_atr"] = (ema_20 - ema_50) / (atr_14 + 1e-8)

    # Сетап на откат: тренд вверх, но идет охлаждение к EMA20
    feats["is_setup"] = ((c > ema_50) & (rsi < 52.0) & (l <= ema_20 * 1.005)).astype(int)

    return feats

def main():
    print("=" * 65)
    print("  СБОРКА ДВУХЛЕТНЕГО FEATURE STORE (2023-2024)")
    print("=" * 65)

    btc_path = DATA_DIR / "BTC_1h.csv"
    if not btc_path.exists():
        print(f"[-] Файл {btc_path} не найден!")
        sys.exit(1)

    btc_raw = pd.read_csv(btc_path)
    btc_feats = compute_single_asset_features(btc_raw, "BTC")

    btc_c = btc_feats["close"]
    btc_atr = btc_feats["atr_14"]
    btc_ret_1h = btc_c - btc_c.shift(1)
    btc_ret_4h_pct = (btc_c - btc_c.shift(4)) / btc_c.shift(4)

    flash_dump = (btc_ret_1h < -1.5 * btc_atr) | (btc_ret_4h_pct < -0.025)
    btc_bull = (btc_feats["trend_spread_atr"] > 0) & (~flash_dump)

    master_gate_df = pd.DataFrame({
        "timestamp": btc_feats["timestamp"],
        "btc_log_ret_24h": btc_feats["log_ret_24h"],
        "master_gate_open": btc_bull.astype(int)
    })

    all_coin_dfs = []

    for coin in ["SOL", "NEAR", "SUI", "ETH"]:
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
        atr_arr = merged["atr_14"].to_numpy()
        ema20_arr = merged["ema_20"].to_numpy()
        setup_arr = merged["is_setup"].to_numpy()
        mg_arr = merged["master_gate_open"].to_numpy()
        rs_arr = merged["relative_strength_24h"].to_numpy()

        n = len(merged)
        labels = np.full(n, np.nan)

        # Разметка касания EMA20 и исхода сделки (TP/SL)
        for i in range(n - TIMEOUT_BARS - 2):
            if setup_arr[i] == 1 and mg_arr[i] == 1 and rs_arr[i] > 0:
                limit_p = ema20_arr[i]
                cur_atr = atr_arr[i]

                # Проверяем исполнение лимитки на следующих 2 барах
                filled = False
                fill_idx = -1
                for f_bar in [1, 2]:
                    if l_arr[i + f_bar] <= limit_p <= h_arr[i + f_bar]:
                        filled = True
                        fill_idx = i + f_bar
                        break

                if not filled:
                    continue

                sl_p = limit_p - (cur_atr * SL_MULT)
                tp_p = limit_p + (cur_atr * TP_MULT)

                outcome = 0
                for k in range(1, TIMEOUT_BARS + 1):
                    bar = fill_idx + k
                    if bar >= n:
                        break
                    if l_arr[bar] <= sl_p:
                        outcome = 0
                        break
                    if h_arr[bar] >= tp_p:
                        outcome = 1
                        break
                labels[i] = outcome

        merged["meta_label"] = labels
        clean_coin = merged.dropna(subset=["atr_14", "vol_zscore_72"]).reset_index(drop=True)
        all_coin_dfs.append(clean_coin)

        valid_setups = clean_coin["meta_label"].dropna()
        win_rate = (valid_setups.sum() / len(valid_setups)) * 100.0 if len(valid_setups) > 0 else 0
        print(f"[+] {coin:<5}: {len(clean_coin):,} баров | Исполненных сетапов: {len(valid_setups):<5} | Базовый WinRate: {win_rate:.1f}%")

    universe_df = pd.concat(all_coin_dfs, ignore_index=True)
    universe_df = universe_df.sort_values(["timestamp", "coin"]).reset_index(drop=True)

    save_path = DATA_DIR / "deep_features_pooled.parquet"
    universe_df.to_parquet(save_path, index=False)

    total_valid = universe_df["meta_label"].dropna()
    print("=" * 65)
    print(f"Всего строк в 2-летнем датасете : {len(universe_df):,}")
    print(f"Всего валидных откатных входов  : {len(total_valid):,}")
    print(f"Файл сохранен в                 : {save_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
