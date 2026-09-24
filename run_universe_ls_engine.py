#!/usr/bin/env python3
"""
Institutional Long/Short Ranking Engine for Hyperliquid Universe.
Trades both directions on high-beta leaders (SOL, SUI, HYPE, NEAR, ETH).
Strict $40 (0.40%) risk per trade and $200 daily loss limit.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

PARQUET_PATH = Path("data/universe/universe_ls_pooled.parquet")
if not PARQUET_PATH.exists():
    print(f"[-] Файл {PARQUET_PATH} не найден!")
    sys.exit(1)

df = pd.read_parquet(PARQUET_PATH)
df = df.sort_values(["timestamp", "coin"]).reset_index(drop=True)

# Исключаем DOGE, XRP, UNI (оставляем фокусных моментум-лидеров)
ACTIVE_UNIVERSE = ["SOL", "SUI", "HYPE", "NEAR", "ETH"]
df = df[df["coin"].isin(ACTIVE_UNIVERSE)].copy().reset_index(drop=True)

FEATURE_COLS = [
    "ret_1h_atr", "ret_4h_atr", "ret_12h_atr", "ret_24h_atr",
    "vol_zscore_72", "rsi_norm", "bollinger_pct_b",
    "vol_compression", "bar_pressure", "trend_spread_atr",
    "relative_strength_24h"
]

TIMEOUT_BARS = 24
SL_MULT = 1.30
TP1_MULT = 1.20
TP2_MULT = 2.20
FEE_PCT = 0.0012
RISK_USD = 40.0
INITIAL_CAPITAL = 10000.0

# Временное разделение Train / OOS
unique_timestamps = np.sort(df["timestamp"].unique())
n_timestamps = len(unique_timestamps)
split_idx = int(n_timestamps * 0.65)
train_cutoff_ts = unique_timestamps[split_idx]
test_start_ts = train_cutoff_ts + ((TIMEOUT_BARS + 1) * 3600 * 1000)

train_mask = (df["timestamp"] <= train_cutoff_ts) & (df["side"] != 0) & (~df["meta_label"].isna())
test_setup_mask = (df["timestamp"] >= test_start_ts) & (df["side"] != 0) & (~df["meta_label"].isna())

X_train = df.loc[train_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
y_train = df.loc[train_mask, "meta_label"].to_numpy(dtype=np.int32)

print(f"[+] Обучающая выборка (SOL, SUI, HYPE, NEAR, ETH): {len(X_train):,} сетапов")
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

# Скоринг: для лонга берем сильные (высокий RS), для шорта — слабые (низкий RS)
df["ranking_score"] = np.where(
    df["side"] == 1,
    0.40 * df["pred_prob"] + 0.60 * df["relative_strength_24h"],
    0.40 * df["pred_prob"] - 0.60 * df["relative_strength_24h"]
)

test_df = df[df["timestamp"] >= test_start_ts].copy().reset_index(drop=True)
unique_test_ts = np.sort(test_df["timestamp"].unique())

CONF_THRESHOLD = 0.55

equity = INITIAL_CAPITAL
peak_equity = equity
equity_curve = [equity]
trades = []
open_positions = []

daily_pnl = 0.0
cur_day = None
day_locked = False
system_halted = False

test_grouped = {ts: g for ts, g in test_df.groupby("timestamp")}

print(f"[*] Запуск двустороннего OOS симулятора ({len(unique_test_ts):,} баров)...")

for ts_idx, cur_ts in enumerate(unique_test_ts):
    if cur_ts not in test_grouped:
        equity_curve.append(equity)
        continue

    bar_df = test_grouped[cur_ts]
    cur_datetime = bar_df["datetime"].iloc[0]
    bar_day = cur_datetime.date() if hasattr(cur_datetime, "date") else str(cur_datetime)[:10]

    if bar_day != cur_day:
        cur_day = bar_day
        daily_pnl = 0.0
        day_locked = False

    if system_halted:
        equity_curve.append(equity)
        continue

    # Обновление открытых позиций
    closed_positions = []
    for pos in open_positions:
        coin = pos["coin"]
        side = pos["side"]
        row = bar_df[bar_df["coin"] == coin]
        if row.empty:
            pos["dur"] += 1
            continue

        cur_h = row["high"].values[0]
        cur_l = row["low"].values[0]
        cur_c = row["close"].values[0]
        pos["dur"] += 1

        # 1. Проверка TP1
        if not pos["tp1_hit"]:
            hit_tp1 = (cur_h >= pos["tp1_price"]) if side == 1 else (cur_l <= pos["tp1_price"])
            if hit_tp1:
                pos["tp1_hit"] = True
                ret1 = (pos["tp1_price"] / pos["entry_price"] - 1.0) if side == 1 else (1.0 - pos["tp1_price"] / pos["entry_price"])
                pnl1 = (pos["pos_usd"] * 0.5) * (ret1 - FEE_PCT)
                equity += pnl1
                daily_pnl += pnl1
                pos["pnl_accum"] += pnl1
                # Перевод стопа оставшейся половины в безубыток
                pos["sl_price"] = pos["entry_price"] * (1.0 + (FEE_PCT if side == 1 else -FEE_PCT))

        # 2. Проверка выхода
        exit_remaining = False
        exit_p = cur_c
        reason = ""

        hit_sl = (cur_l <= pos["sl_price"]) if side == 1 else (cur_h >= pos["sl_price"])
        hit_tp2 = (cur_h >= pos["tp2_price"]) if side == 1 else (cur_l <= pos["tp2_price"])

        if hit_sl:
            exit_p = pos["sl_price"]
            reason = "SL_BE" if pos["tp1_hit"] else "SL_FULL"
            exit_remaining = True
        elif pos["tp1_hit"] and hit_tp2:
            exit_p = pos["tp2_price"]
            reason = "TP2_WIN"
            exit_remaining = True
        elif pos["dur"] >= TIMEOUT_BARS:
            exit_p = cur_c
            reason = "TIMEOUT"
            exit_remaining = True

        if exit_remaining:
            rem_fraction = 0.5 if pos["tp1_hit"] else 1.0
            ret2 = (exit_p / pos["entry_price"] - 1.0) if side == 1 else (1.0 - exit_p / pos["entry_price"])
            pnl2 = (pos["pos_usd"] * rem_fraction) * (ret2 - FEE_PCT)
            equity += pnl2
            daily_pnl += pnl2
            pos["pnl_accum"] += pnl2

            trades.append({
                "coin": coin,
                "side": "LONG" if side == 1 else "SHORT",
                "entry_time": pos["entry_time"],
                "exit_time": cur_datetime,
                "entry": pos["entry_price"],
                "exit": exit_p,
                "pnl": pos["pnl_accum"],
                "reason": reason
            })
            closed_positions.append(pos)

            if daily_pnl <= -200.0:
                day_locked = True

    for pos in closed_positions:
        open_positions.remove(pos)

    # Мониторинг просадки
    peak_equity = max(peak_equity, equity)
    drawdown = (peak_equity - equity) / peak_equity
    if drawdown >= 0.08:
        print(f"[!] АВАРИЙНЫЙ СТОП: Достигнут лимит просадки {drawdown*100:.2f}%!")
        system_halted = True

    equity_curve.append(equity)

    # Селекция новых сделок через Ranking Engine
    can_enter = (
        not day_locked and not system_halted and
        len(open_positions) < 2 and
        (len(open_positions) == 0 or open_positions[0]["tp1_hit"])
    )

    if can_enter and ts_idx + 1 < len(unique_test_ts):
        next_ts = unique_test_ts[ts_idx + 1]
        next_bar_df = test_grouped.get(next_ts, None)

        if next_bar_df is not None:
            candidates = bar_df[
                (bar_df["side"] != 0) &
                (bar_df["pred_prob"] >= CONF_THRESHOLD)
            ]

            active_coins = {p["coin"] for p in open_positions}
            candidates = candidates[~candidates["coin"].isin(active_coins)]

            if not candidates.empty:
                best_candidate = candidates.sort_values("ranking_score", ascending=False).iloc[0]
                best_coin = best_candidate["coin"]
                best_side = int(best_candidate["side"])

                next_coin_row = next_bar_df[next_bar_df["coin"] == best_coin]
                if not next_coin_row.empty:
                    entry_price = next_coin_row["open"].values[0]
                    cur_atr = best_candidate["atr_14"]

                    sl_dist_pct = (cur_atr * SL_MULT) / entry_price
                    pos_size_usd = min(RISK_USD / (sl_dist_pct + 1e-8), equity * 1.5)

                    if best_side == 1:
                        sl_p = entry_price - (cur_atr * SL_MULT)
                        tp1_p = entry_price + (cur_atr * TP1_MULT)
                        tp2_p = entry_price + (cur_atr * TP2_MULT)
                    else:
                        sl_p = entry_price + (cur_atr * SL_MULT)
                        tp1_p = entry_price - (cur_atr * TP1_MULT)
                        tp2_p = entry_price - (cur_atr * TP2_MULT)

                    open_positions.append({
                        "coin": best_coin,
                        "side": best_side,
                        "entry_time": next_coin_row["datetime"].values[0],
                        "entry_price": entry_price,
                        "pos_usd": pos_size_usd,
                        "sl_price": sl_p,
                        "tp1_price": tp1_p,
                        "tp2_price": tp2_p,
                        "tp1_hit": False,
                        "dur": 0,
                        "pnl_accum": 0.0
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
trades_per_month = (len(df_trades) / (total_days / 30.0)) if total_days > 0 else 0.0

print("=" * 65)
print("  РЕЗУЛЬТАТЫ ДВУСТОРОННЕГО БЭКТЕСТА LONG/SHORT (OOS)")
print("=" * 65)
print(f"Период тестирования   : {total_days:.1f} дней (~{total_days/30.0:.1f} мес.)")
print(f"Начальный баланс      : ${INITIAL_CAPITAL:,.2f}")
print(f"Конечный баланс       : ${equity:,.2f}")
print(f"Чистый PnL            : {total_pnl:+.2f}%")
print(f"Максимальная просадка : -{max_dd:.2f}% (Лимит: < 8.0%)")
print(f"Всего сделок          : {len(df_trades)} (~{trades_per_month:.1f} сделок/мес)")
print(f"Win Rate              : {wr:.1f}%")
print(f"Profit Factor         : {pf:.3f}")

if not df_trades.empty:
    print("\nРаспределение по сторонам:")
    print(df_trades["side"].value_counts().to_string())
    print("\nРаспределение по инструментам:")
    print(df_trades["coin"].value_counts().to_string())
    print("\nПричины выходов:")
    print(df_trades["reason"].value_counts().to_string())
print("=" * 65)

# Сохранение результатов и графика
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
if not df_trades.empty:
    df_trades.to_csv(results_dir / "universe_ls_trades.csv", index=False)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(eq_arr, label="Portfolio Equity ($)", color="#1f77b4", lw=1.5)
plt.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.7)
plt.title("Hyperliquid Long/Short Portfolio Equity (OOS)")
plt.ylabel("Balance ($)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(-dds * 100.0, label="Drawdown (%)", color="#d62728", lw=1.2)
plt.axhline(-8.0, color="black", linestyle="--", label="Prop Limit (-8.0%)")
plt.ylabel("Drawdown (%)")
plt.xlabel("Hourly Bars")
plt.grid(True, alpha=0.3)
plt.legend()

chart_path = results_dir / "universe_ls_chart.png"
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"[+] График сохранен в: {chart_path}")
