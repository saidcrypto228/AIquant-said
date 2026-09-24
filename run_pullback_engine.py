#!/usr/bin/env python3
"""
Trend-Pullback Engine with Maker Limit Execution & Asymmetric R:R (2.4:1).
Universe: SOL, SUI, HYPE, NEAR (BTC Master Gate).
Sizing: 1.5% equity risk ($3.00 on $200).
Cooldown: 12h per asset post-SL.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

PARQUET_PATH = Path("data/universe/universe_pooled_features.parquet")
if not PARQUET_PATH.exists():
    print(f"[-] Файл {PARQUET_PATH} не найден!")
    sys.exit(1)

df = pd.read_parquet(PARQUET_PATH)
df = df.sort_values(["timestamp", "coin"]).reset_index(drop=True)

# Торгуем исключительно топ-лидеров
CORE_UNIVERSE = ["SOL", "SUI", "HYPE", "NEAR"]
df = df[df["coin"].isin(CORE_UNIVERSE)].copy().reset_index(drop=True)

FEATURE_COLS = [
    "ret_1h_atr", "ret_4h_atr", "ret_12h_atr", "ret_24h_atr",
    "vol_zscore_72", "rsi_norm", "bollinger_pct_b",
    "vol_compression", "bar_pressure", "trend_spread_atr",
    "relative_strength_24h"
]

TIMEOUT_BARS = 24
SL_MULT = 0.90          # Сжатый стоп на откате (0.9 ATR)
TP_MULT = 2.20          # Целевой тейк (2.2 ATR) -> RR = 2.44:1
FEE_MAKER_PCT = 0.0002  # Лимитный вход (Maker fee на Hyperliquid: 0.02% или рибейт)
FEE_TAKER_PCT = 0.0005  # Выход по стопу/тейку

INITIAL_CAPITAL = 200.0 # Реалистичный депозит $200
RISK_PCT = 0.015        # 1.5% риска на сделку ($3.00)
COOLDOWN_HOURS = 12

unique_timestamps = np.sort(df["timestamp"].unique())
n_timestamps = len(unique_timestamps)
split_idx = int(n_timestamps * 0.65)
train_cutoff_ts = unique_timestamps[split_idx]
test_start_ts = train_cutoff_ts + ((TIMEOUT_BARS + 1) * 3600 * 1000)

train_mask = (df["timestamp"] <= train_cutoff_ts) & (df["is_setup"] == 1) & (~df["meta_label"].isna())
test_setup_mask = (df["timestamp"] >= test_start_ts) & (df["is_setup"] == 1) & (~df["meta_label"].isna())

X_train = df.loc[train_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
y_train = df.loc[train_mask, "meta_label"].to_numpy(dtype=np.int32)

print(f"[+] Обучающая выборка: {len(X_train):,} сетапов")
print(f"[+] Тестовая OOS выборка: {test_setup_mask.sum():,} сетапов")

pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

meta_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    tree_method="hist"
)
meta_clf.fit(X_train, y_train)

all_X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
df["pred_prob"] = meta_clf.predict_proba(all_X)[:, 1]
df["ranking_score"] = 0.40 * df["pred_prob"] + 0.60 * df["relative_strength_24h"]

# Считаем часовую EMA20 для лимитных уровней входа
df["ema_20"] = df.groupby("coin")["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())

test_df = df[df["timestamp"] >= test_start_ts].copy().reset_index(drop=True)
unique_test_ts = np.sort(test_df["timestamp"].unique())

equity = INITIAL_CAPITAL
peak_equity = equity
equity_curve = [equity]
trades = []
open_positions = []
pending_orders = [] # Очередь выставленных лимитных ордеров

asset_cooldown = {} # Заморозка монет после стопа

test_grouped = {ts: g for ts, g in test_df.groupby("timestamp")}

print(f"[*] Симуляция лимитных входов на откатах ({len(unique_test_ts):,} баров)...")

for ts_idx, cur_ts in enumerate(unique_test_ts):
    if cur_ts not in test_grouped:
        equity_curve.append(equity)
        continue

    bar_df = test_grouped[cur_ts]
    cur_datetime = bar_df["datetime"].iloc[0]

    # 1. Проверка исполнения лимитных ордеров (Pending Limit Orders)
    still_pending = []
    for order in pending_orders:
        coin = order["coin"]
        row = bar_df[bar_df["coin"] == coin]
        if row.empty:
            order["ttl"] -= 1
            if order["ttl"] > 0:
                still_pending.append(order)
            continue

        cur_l = row["low"].values[0]
        cur_h = row["high"].values[0]
        limit_p = order["limit_price"]

        # Лимитный ордер исполнился, если цена опустилась до EMA20
        if cur_l <= limit_p <= cur_h and len(open_positions) == 0:
            atr = order["atr"]
            sl_price = limit_p - (atr * SL_MULT)
            tp_price = limit_p + (atr * TP_MULT)

            # Расчет сайзинга под риск 1.5%
            risk_usd = equity * RISK_PCT
            sl_dist_pct = (limit_p - sl_price) / limit_p
            pos_usd = max(risk_usd / sl_dist_pct, 12.0) # Не менее $12 для Hyperliquid

            open_positions.append({
                "coin": coin,
                "entry_time": cur_datetime,
                "entry_price": limit_p,
                "pos_usd": pos_usd,
                "sl_price": sl_price,
                "tp_price": tp_price,
                "dur": 0
            })
        else:
            order["ttl"] -= 1
            if order["ttl"] > 0:
                still_pending.append(order)

    pending_orders = still_pending

    # 2. Сопровождение открытых позиций
    closed_positions = []
    for pos in open_positions:
        coin = pos["coin"]
        row = bar_df[bar_df["coin"] == coin]
        if row.empty:
            pos["dur"] += 1
            continue

        cur_h = row["high"].values[0]
        cur_l = row["low"].values[0]
        cur_c = row["close"].values[0]
        pos["dur"] += 1

        exit_p = None
        reason = ""

        if cur_l <= pos["sl_price"]:
            exit_p = pos["sl_price"]
            reason = "SL_TIGHT"
            # Активируем заморозку актива на 12 часов
            asset_cooldown[coin] = cur_ts + (COOLDOWN_HOURS * 3600 * 1000)
        elif cur_h >= pos["tp_price"]:
            exit_p = pos["tp_price"]
            reason = "TP_FULL"
        elif pos["dur"] >= TIMEOUT_BARS:
            exit_p = cur_c
            reason = "TIMEOUT"

        if exit_p is not None:
            raw_ret = (exit_p / pos["entry_price"]) - 1.0
            fees = FEE_MAKER_PCT + FEE_TAKER_PCT
            pnl_usd = pos["pos_usd"] * (raw_ret - fees)

            equity += pnl_usd
            trades.append({
                "coin": coin,
                "entry_time": pos["entry_time"],
                "exit_time": cur_datetime,
                "entry": pos["entry_price"],
                "exit": exit_p,
                "pnl": pnl_usd,
                "reason": reason
            })
            closed_positions.append(pos)

    for pos in closed_positions:
        open_positions.remove(pos)

    equity_curve.append(equity)

    # 3. Генерация новых лимитных ордеров
    can_place = (len(open_positions) == 0 and len(pending_orders) == 0)

    if can_place:
        # Отбор кандидатов
        candidates = bar_df[
            (bar_df["master_gate_open"] == 1) &
            (bar_df["btc_log_ret_24h"] > 0) &
            (bar_df["is_setup"] == 1) &
            (bar_df["pred_prob"] >= 0.55) &
            (bar_df["relative_strength_24h"] > 0.01)
        ]

        # Фильтр кулдауна
        valid_candidates = []
        for idx_c, row_c in candidates.iterrows():
            c_coin = row_c["coin"]
            if asset_cooldown.get(c_coin, 0) <= cur_ts:
                valid_candidates.append(row_c)

        if len(valid_candidates) > 0:
            df_valid = pd.DataFrame(valid_candidates)
            best = df_valid.sort_values("ranking_score", ascending=False).iloc[0]

            # Выставляем лимит на уровне EMA20 со сроком жизни 2 бара
            pending_orders.append({
                "coin": best["coin"],
                "limit_price": best["ema_20"],
                "atr": best["atr_14"],
                "ttl": 2
            })

df_trades = pd.DataFrame(trades)
eq_arr = np.array(equity_curve)
peaks = np.maximum.accumulate(eq_arr)
dds = (peaks - eq_arr) / peaks
max_dd = dds.max() * 100.0
total_pnl = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

win_trades = df_trades[df_trades["pnl"] > 0] if not df_trades.empty else pd.DataFrame()
loss_trades = df_trades[df_trades["pnl"] < 0] if not df_trades.empty else pd.DataFrame()
wr = (len(win_trades) / len(df_trades)) * 100.0 if not df_trades.empty else 0.0
pf = (win_trades["pnl"].sum() / abs(loss_trades["pnl"].sum())) if not loss_trades.empty else 0.0

total_days = len(unique_test_ts) / 24.0

print("=" * 65)
print("  РЕЗУЛЬТАТЫ СИМУЛЯТОРА TREND-PULLBACK (ЛИМИТНЫЕ ВХОДЫ)")
print("=" * 65)
print(f"Период тестирования   : {total_days:.1f} дней (~{total_days/30.0:.1f} мес.)")
print(f"Стартовый депозит     : ${INITIAL_CAPITAL:,.2f}")
print(f"Конечный баланс       : ${equity:,.2f}")
print(f"Чистый PnL            : {total_pnl:+.2f}%")
print(f"Максимальная просадка : -{max_dd:.2f}%")
print(f"Всего сделок          : {len(df_trades)}")
print(f"Win Rate              : {wr:.1f}%")
print(f"Profit Factor         : {pf:.3f}")

if not df_trades.empty:
    print("\nРаспределение по инструментам:")
    print(df_trades["coin"].value_counts().to_string())
    print("\nПричины выходов:")
    print(df_trades["reason"].value_counts().to_string())
    print("\nПервые 5 сделок:")
    print(df_trades[["coin", "entry", "exit", "pnl", "reason"]].head().to_string())
print("=" * 65)
