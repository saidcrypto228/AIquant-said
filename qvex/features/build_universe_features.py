#!/usr/bin/env python3
"""
Cross-Sectional Feature Store & Master Gate Pipeline for Hyperliquid Universe.
Combines 9 liquid contracts into an invariant, normalized panel dataset.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/universe")
TARGET_COINS = ["BTC", "ETH", "SOL", "HYPE", "NEAR", "SUI", "XRP", "DOGE", "UNI"]

SL_MULT = 1.30
TP1_MULT = 1.20
TIMEOUT_BARS = 36

def compute_single_asset_features(df: pd.DataFrame, coin: str) -> pd.DataFrame:
    df = df.copy().sort_values("timestamp").reset_index(drop=True)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # 1. Волатильность (True Range & ATR)
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    atr_fast = tr.rolling(window=6, min_periods=6).mean()
    atr_slow = tr.rolling(window=48, min_periods=48).mean()
    atr_norm = atr_14 / c

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

    # 2. Доходности, приведенные к волатильности (ATR-Normalized Returns)
    feats["ret_1h_atr"] = (c - c.shift(1)) / (atr_14 + 1e-8)
    feats["ret_4h_atr"] = (c - c.shift(4)) / (atr_14 + 1e-8)
    feats["ret_12h_atr"] = (c - c.shift(12)) / (atr_14 + 1e-8)
    feats["ret_24h_atr"] = (c - c.shift(24)) / (atr_14 + 1e-8)

    # Сырой 24h лог-возврат для расчета Relative Strength
    feats["log_ret_24h"] = np.log(c / c.shift(24))

    # 3. Z-score объема (72 бара = 3 суток)
    vol_mean = v.rolling(window=72, min_periods=72).mean()
    vol_std = v.rolling(window=72, min_periods=72).std()
    feats["vol_zscore_72"] = ((v - vol_mean) / (vol_std + 1e-8)).clip(-3.0, 5.0)

    # 4. Осцилляторы и структурные метрики
    # RSI(14) -> диапазон [-1, 1]
    delta = c.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    loss = ((-delta.where(delta < 0, 0.0))).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    feats["rsi_norm"] = (rsi - 50.0) / 50.0

    # Bollinger %B (20, 2) -> безразмерное положение в канале
    sma_20 = c.rolling(window=20, min_periods=20).mean()
    std_20 = c.rolling(window=20, min_periods=20).std()
    upper_b = sma_20 + 2.0 * std_20
    lower_b = sma_20 - 2.0 * std_20
    b_range = (upper_b - lower_b).replace(0, np.nan)
    feats["bollinger_pct_b"] = ((c - lower_b) / b_range).clip(-0.5, 1.5)

    # Сжатие волатильности и давление баров
    feats["vol_compression"] = (atr_fast / (atr_slow + 1e-8)).clip(0.2, 3.0)
    range_safe = (h - l).replace(0, np.nan)
    feats["bar_pressure"] = ((2 * c - h - l) / range_safe).fillna(0.0)

    # Трендовый спред
    ema_20 = c.ewm(span=20, adjust=False).mean()
    ema_50 = c.ewm(span=50, adjust=False).mean()
    feats["trend_spread_atr"] = (ema_20 - ema_50) / (atr_14 + 1e-8)

    # Вспомогательные уровни для фильтра первичного сетапа
    feats["c_gt_ema50"] = (c > ema_50).astype(int)
    feats["pullback_trigger"] = ((rsi < 45.0) | (l <= ema_20)).astype(int)

    return feats

def main():
    print("=" * 65)
    print("  ПОСТРОЕНИЕ КРОСС-СЕКЦИОННОГО FEATURE STORE")
    print("=" * 65)

    # 1. Загрузка BTC для формирования Master Gate
    btc_path = DATA_DIR / "BTC_1h.csv"
    if not btc_path.exists():
        print(f"[-] Файл {btc_path} не найден!")
        sys.exit(1)

    print("[*] Расчет макро-режима Master Gate на BTC...")
    btc_raw = pd.read_csv(btc_path)
    btc_feats = compute_single_asset_features(btc_raw, "BTC")

    # Индикаторы шока на BTC
    btc_c = btc_feats["close"]
    btc_atr = btc_feats["atr_14"]
    btc_ret_1h = btc_c - btc_c.shift(1)
    btc_ret_4h_pct = (btc_c - btc_c.shift(4)) / btc_c.shift(4)

    # Правила блокировки Master Gate
    flash_dump = (btc_ret_1h < -1.5 * btc_atr) | (btc_ret_4h_pct < -0.025)
    btc_macro_bull = (btc_feats["c_gt_ema50"] == 1) & (btc_feats["trend_spread_atr"] > 0)

    master_gate_df = pd.DataFrame({
        "timestamp": btc_feats["timestamp"],
        "btc_log_ret_24h": btc_feats["log_ret_24h"],
        "master_gate_open": (btc_macro_bull & (~flash_dump)).astype(int),
        "btc_flash_dump": flash_dump.astype(int)
    })

    # 2. Обработка всех монет корзины
    all_coin_dfs = []

    for coin in TARGET_COINS:
        file_path = DATA_DIR / f"{coin}_1h.csv"
        if not file_path.exists():
            print(f"[!] Предупреждение: {coin} отсутствует, пропуск.")
            continue

        raw = pd.read_csv(file_path)
        coin_feats = compute_single_asset_features(raw, coin)

        # Слияние с Master Gate по временной метке
        merged = pd.merge(coin_feats, master_gate_df, on="timestamp", how="inner")

        # Расчет Relative Strength к BTC
        merged["relative_strength_24h"] = merged["log_ret_24h"] - merged["btc_log_ret_24h"]

        # Разметка Meta-Labels (Triple Barrier)
        c_arr = merged["close"].to_numpy()
        h_arr = merged["high"].to_numpy()
        l_arr = merged["low"].to_numpy()
        o_arr = merged["open"].to_numpy()
        atr_arr = merged["atr_14"].to_numpy()
        mg_open = merged["master_gate_open"].to_numpy()
        pullback = merged["pullback_trigger"].to_numpy()
        c_gt_ema = merged["c_gt_ema50"].to_numpy()

        n_rows = len(merged)
        labels = np.full(n_rows, np.nan)
        setups = np.zeros(n_rows, dtype=int)

        # Первичный сетап: собственный аптренд + откат + Master Gate открыт
        is_setup = (c_gt_ema == 1) & (pullback == 1) & (mg_open == 1)

        for i in range(n_rows - TIMEOUT_BARS - 1):
            if not is_setup[i]:
                continue

            setups[i] = 1
            entry = o_arr[i + 1]
            cur_atr = atr_arr[i]
            sl_price = entry - (cur_atr * SL_MULT)
            tp1_price = entry + (cur_atr * TP1_MULT)

            success = 0
            for k in range(1, TIMEOUT_BARS + 1):
                idx = i + k
                if l_arr[idx] <= sl_price:
                    success = 0
                    break
                if h_arr[idx] >= tp1_price:
                    success = 1
                    break
            labels[i] = success

        merged["is_setup"] = setups
        merged["meta_label"] = labels

        # Удаляем строки с неполными скользящими окнами
        clean_coin = merged.dropna(subset=["atr_14", "vol_zscore_72", "relative_strength_24h"]).reset_index(drop=True)
        all_coin_dfs.append(clean_coin)
        print(f"[+] {coin:<5}: {len(clean_coin):,} баров обработано | Сетапов: {clean_coin['is_setup'].sum():,}")

    # 3. Объединение в кросс-секционную панель
    universe_df = pd.concat(all_coin_dfs, ignore_index=True)
    # Сортировка по времени для предотвращения перемешивания хронологии
    universe_df = universe_df.sort_values(["timestamp", "coin"]).reset_index(drop=True)

    save_path = DATA_DIR / "universe_pooled_features.parquet"
    universe_df.to_parquet(save_path, index=False)

    csv_sample_path = DATA_DIR / "universe_pooled_sample.csv"
    universe_df.head(100).to_csv(csv_sample_path, index=False)

    total_setups = universe_df['is_setup'].sum()
    valid_labels = universe_df['meta_label'].dropna()
    pos_rate = (valid_labels == 1).mean() * 100.0 if len(valid_labels) > 0 else 0.0

    print("=" * 65)
    print("  ИТОГИ ФОРМИРОВАНИЯ ПАНЕЛЬНОГО ДАТАСЕТА:")
    print("=" * 65)
    print(f"Всего строк в панели      : {len(universe_df):,}")
    print(f"Всего валидных сетапов    : {total_setups:,}")
    print(f"Базовый Win Rate (TP1/SL) : {pos_rate:.1f}%")
    print(f"Файл сохранен в           : {save_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
