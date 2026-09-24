#!/usr/bin/env python3
"""
Институциональный конвейер ML Meta-Labeling v9.2 (Ridge Logistic Regression).
- Замена оверфитящегося бустинга на L2-регуляризованную логистическую регрессию.
- Стандартизация признаков (StandardScaler) строго на обучающей выборке.
- Purged Split с 24-часовым эмбарго де Прадо.
- Экспорт весов и параметров скейлера в открытый JSON data/meta_model.json.
"""

import time
import math
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from hyperliquid.info import Info
from hyperliquid.utils import constants

import bot_config as config
from quant_factors import QuantFactorEngine

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, precision_score
except ImportError:
    print("[-] Ошибка: scikit-learn не установлен. Выполните: pip install scikit-learn")
    exit(1)

print("=" * 85)
print("  ФАЗА 5: ОБУЧЕНИЕ РЕГУЛЯРИЗОВАННОЙ META-МОДЕЛИ (RIDGE LOGISTIC v9.2)")
print("=" * 85)

clean_target_coins = [
    c for c in config.TARGET_COINS 
    if c not in ["ETH", "LINK", "PEPE", "kPEPE", "WIF"]
]

base_url = constants.TESTNET_API_URL if config.IS_TESTNET else constants.MAINNET_API_URL
info = Info(base_url, skip_ws=True)

END_MS = int(time.time() * 1000)
START_MS = END_MS - (1500 * 3600 * 1000)
SYMBOLS = list(set(["BTC"] + clean_target_coins))

print(f"[*] Сбор исторических данных Hyperliquid ({len(SYMBOLS)} инструментов)...")
data_1h: Dict[str, pd.DataFrame] = {}

for sym in SYMBOLS:
    try:
        raw = info.candles_snapshot(name=sym, interval="1h", startTime=START_MS, endTime=END_MS)
        if raw and len(raw) >= 200:
            df = pd.DataFrame([{
                "t": int(c["t"]),
                "dt": pd.to_datetime(c["t"], unit="ms", utc=True),
                "open": float(c["o"]),
                "high": float(c["h"]),
                "low": float(c["l"]),
                "close": float(c["c"]),
                "vol": float(c["v"])
            } for c in raw]).sort_values("t").reset_index(drop=True)
            data_1h[sym] = df
    except Exception:
        pass

print(f"[✓] Загружено проверенных активов: {len(data_1h)}")

# Подготовка 4H таймфрейма со строгим сдвигом
processed_1h: Dict[str, pd.DataFrame] = {}
for sym, df in data_1h.items():
    df = df.copy()
    df["vol_rolling_4h"] = df["vol"].rolling(4, min_periods=4).sum()

    df_temp = df.set_index("dt")
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "vol": "sum", "t": "first"}
    df_4h = df_temp.resample("4h", label="left", closed="left").agg(ohlc).dropna().reset_index()

    c = df_4h["close"]
    h = df_4h["high"]
    l = df_4h["low"]
    v = df_4h["vol"]
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)

    df_4h["atr_4h"] = tr.rolling(14, min_periods=14).mean()
    df_4h["ema20_4h"] = c.ewm(span=20, adjust=False).mean()
    df_4h["ema50_4h"] = c.ewm(span=50, adjust=False).mean()
    df_4h["ema50_slope"] = df_4h["ema50_4h"] - df_4h["ema50_4h"].shift(3)
    df_4h["donchian_high_4h"] = h.rolling(36).max()
    df_4h["donchian_low_4h"] = l.rolling(36).min()
    df_4h["vol_sma20_4h"] = v.rolling(20, min_periods=20).mean()

    # Сдвиг на 1 бар назад
    df_4h_shifted = df_4h[[
        "dt", "atr_4h", "ema20_4h", "ema50_4h", "ema50_slope",
        "donchian_high_4h", "donchian_low_4h", "vol_sma20_4h"
    ]].copy()

    for col in ["atr_4h", "ema20_4h", "ema50_4h", "ema50_slope", "donchian_high_4h", "donchian_low_4h", "vol_sma20_4h"]:
        df_4h_shifted[col] = df_4h_shifted[col].shift(1)

    merged = pd.merge_asof(
        df.sort_values("dt"),
        df_4h_shifted.sort_values("dt"),
        on="dt",
        direction="backward"
    ).dropna().reset_index(drop=True)

    processed_1h[sym] = merged

common_timestamps = processed_1h["BTC"]["t"].values
min_warmup = 160

print("[*] Сбор датасета с разметкой от триггера исполнения...")
dataset_records = []
btc_df = processed_1h["BTC"]

for i in range(min_warmup, len(common_timestamps) - 48):
    curr_t = common_timestamps[i]
    btc_row = btc_df[btc_df["t"] == curr_t]
    if btc_row.empty:
        continue
    btc_row = btc_row.iloc[0]

    btc_bull = bool(btc_row["close"] > btc_row["ema50_4h"])
    btc_bear = bool(btc_row["close"] < btc_row["ema50_4h"])
    btc_closes = btc_df[btc_df["t"] <= curr_t]["close"]

    for coin in clean_target_coins:
        if coin not in processed_1h:
            continue

        c_df = processed_1h[coin]
        c_sub = c_df[c_df["t"] <= curr_t]
        if len(c_sub) < 72:
            continue

        curr_bar = c_sub.iloc[-1]
        c_closes = c_sub["close"]
        z_res_mom, beta_btc, raw_rs_pct = QuantFactorEngine.compute_residual_momentum_72h(c_closes, btc_closes)

        signal_dir = None
        entry_type = None
        trigger_px = 0.0

        if btc_bull and z_res_mom >= 0.40 and raw_rs_pct >= 2.0:
            vol_boost = curr_bar["vol_rolling_4h"] >= curr_bar["vol_sma20_4h"] * 1.15
            breakout_hit = (curr_bar["close"] >= curr_bar["donchian_high_4h"] * 0.998)

            if breakout_hit and vol_boost and (curr_bar["close"] > curr_bar["ema20_4h"]):
                signal_dir = "LONG"
                entry_type = 1.0
                trigger_px = curr_bar["high"] * 1.0005
            else:
                is_trend = (curr_bar["close"] > curr_bar["ema50_4h"]) and (curr_bar["ema20_4h"] > curr_bar["ema50_4h"])
                is_evr_ok, _, _ = QuantFactorEngine.evaluate_evr_absorption(
                    open_px=curr_bar["open"], high_px=curr_bar["high"], low_px=curr_bar["low"], close_px=curr_bar["close"],
                    volume=curr_bar["vol_rolling_4h"], vol_sma=curr_bar["vol_sma20_4h"], ema20_4h=curr_bar["ema20_4h"], atr_4h=curr_bar["atr_4h"]
                )
                if is_trend and is_evr_ok:
                    signal_dir = "LONG"
                    entry_type = 0.0
                    trigger_px = curr_bar["high"] * 1.0005

        elif btc_bear and z_res_mom <= -0.40 and raw_rs_pct <= -2.0:
            vol_boost = curr_bar["vol_rolling_4h"] >= curr_bar["vol_sma20_4h"] * 1.15
            breakdown_hit = (curr_bar["close"] <= curr_bar["donchian_low_4h"] * 1.002)
            if breakdown_hit and vol_boost and (curr_bar["close"] < curr_bar["ema20_4h"]):
                signal_dir = "SHORT"
                entry_type = 1.0
                trigger_px = curr_bar["low"] * 0.9995

        if signal_dir is None:
            continue

        atr = curr_bar["atr_4h"]
        vol_rel = min(curr_bar["vol_rolling_4h"] / max(curr_bar["vol_sma20_4h"], 1e-4), 5.0)
        dist_ema20 = (curr_bar["close"] - curr_bar["ema20_4h"]) / max(atr, 1e-4)
        donch_range = max(curr_bar["donchian_high_4h"] - curr_bar["donchian_low_4h"], 1e-4)
        donch_pos = (curr_bar["close"] - curr_bar["donchian_low_4h"]) / donch_range
        btc_slope_rel = btc_row["ema50_slope"] / max(btc_row["close"] * 0.01, 1e-4)

        # Проверка фактического исполнения триггера в следующие 3 часа
        future_window = c_df[(c_df["t"] > curr_t) & (c_df["t"] <= curr_t + (3 * 3600 * 1000))]
        if future_window.empty:
            continue

        fill_time = None
        for _, w_row in future_window.iterrows():
            if signal_dir == "LONG" and w_row["high"] >= trigger_px:
                fill_time = w_row["t"]
                break
            elif signal_dir == "SHORT" and w_row["low"] <= trigger_px:
                fill_time = w_row["t"]
                break

        if fill_time is None:
            continue

        # Тройной барьер 48 часов от момента входа
        horizon_bars = c_df[(c_df["t"] > fill_time) & (c_df["t"] <= fill_time + (48 * 3600 * 1000))]
        if len(horizon_bars) < 24:
            continue

        take_barrier = trigger_px + (atr * 1.8) if signal_dir == "LONG" else trigger_px - (atr * 1.8)
        stop_barrier = trigger_px - (atr * 1.5) if signal_dir == "LONG" else trigger_px + (atr * 1.5)

        label = 0
        for _, h_row in horizon_bars.iterrows():
            hit_tp = (h_row["high"] >= take_barrier) if signal_dir == "LONG" else (h_row["low"] <= take_barrier)
            hit_sl = (h_row["low"] <= stop_barrier) if signal_dir == "LONG" else (h_row["high"] >= stop_barrier)

            if hit_tp and hit_sl:
                label = 0  # Консервативное разрешение конфликта
                break
            elif hit_tp:
                label = 1
                break
            elif hit_sl:
                label = 0
                break

        dataset_records.append({
            "t": fill_time,
            "z_res_mom": z_res_mom, "beta_btc": beta_btc, "raw_rs_pct": raw_rs_pct,
            "vol_rel": vol_rel, "atr_pct": (atr / trigger_px) * 100.0,
            "dist_ema20": dist_ema20, "donch_pos": donch_pos,
            "btc_slope_rel": btc_slope_rel, "entry_type": entry_type,
            "is_long": 1.0 if signal_dir == "LONG" else 0.0,
            "label": label
        })

df_ml = pd.DataFrame(dataset_records).sort_values("t").reset_index(drop=True)
print(f"[✓] Датасет сформирован: {len(df_ml)} исполненных сделок.")
print(f"    Баланс классов: {df_ml['label'].value_counts().to_dict()} (Доля успешных: {df_ml['label'].mean()*100:.1f}%)")

# Purged Time-Series Split с 24-часовым эмбарго
test_ratio = 0.25
split_point = int(len(df_ml) * (1.0 - test_ratio))
test_start_t = df_ml.iloc[split_point]["t"]
embargo_ms = 24 * 3600 * 1000

train_mask = df_ml["t"] < (test_start_t - (48 * 3600 * 1000))
test_mask = df_ml["t"] >= (test_start_t + embargo_ms)

feature_cols = [
    "z_res_mom", "beta_btc", "raw_rs_pct", "vol_rel", 
    "atr_pct", "dist_ema20", "donch_pos", "btc_slope_rel", 
    "entry_type", "is_long"
]

X_train = df_ml[train_mask][feature_cols]
y_train = df_ml[train_mask]["label"]
X_test = df_ml[test_mask][feature_cols]
y_test = df_ml[test_mask]["label"]

print(f"[*] Сэмплов в Train: {len(X_train)} | Test (Purged OOS): {len(X_test)}")

# 1. Стандартизация признаков строго по Train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Обучение Ridge Logistic Regression (L2 Penalty)
model = LogisticRegression(
    penalty="l2",
    C=0.25,                 # Умеренная L2-регуляризация против переобучения
    solver="lbfgs",
    max_iter=500,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 3. Валидация на чистом Out-of-Sample
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
y_pred_bin = (y_pred_prob >= 0.50).astype(int)

auc_val = roc_auc_score(y_test, y_pred_prob)
precision_val = precision_score(y_test, y_pred_bin, zero_division=0)

print("\n" + "=" * 85)
print("  РЕЗУЛЬТАТЫ НЕЗАВИСИМОЙ ОЦЕНКИ RIDGE LOGISTIC (PURGED OOS)")
print("=" * 85)
print(f"📊 Честный ROC-AUC:    {auc_val:.3f} (Цель: > 0.52)")
print(f"🎯 Честный Precision:  {precision_val*100:.1f}%")
print("-" * 85)

# Экспорт коэффициентов модели
weights = {
    "feature_cols": feature_cols,
    "coef": model.coef_[0].tolist(),
    "intercept": float(model.intercept_[0]),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_scale": scaler.scale_.tolist(),
    "auc_test": float(auc_val),
    "trained_at": time.time()
}

print("🏆 Веса линейных факторов (Feature Weights):")
for col, w in zip(feature_cols, model.coef_[0]):
    print(f"  > {col:<15}: {w:+.4f}")

model_file = config.DATA_DIR / "meta_model.json"
with open(model_file, "w", encoding="utf-8") as f:
    json.dump(weights, f, indent=2)

print(f"\n[✓] Модель и скейлер сохранены в: {model_file}")
print("=" * 85)
