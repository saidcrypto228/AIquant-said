#!/usr/bin/env python3
"""
Purged Rolling Walk-Forward Validator (2023-2024).
Train window: 6 months (~4380h).
Purge gap: 24h.
Test window: 2 months (~1460h).
Sizing: 1.5% equity risk ($200 start). Limit Maker entry at EMA20.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

DATA_PATH = Path("data/history_deep/deep_features_pooled.parquet")
if not DATA_PATH.exists():
    print(f"[-] Файл {DATA_PATH} не найден!")
    sys.exit(1)

df = pd.read_parquet(DATA_PATH)
df = df.sort_values(["timestamp", "coin"]).reset_index(drop=True)

FEATURE_COLS = [
    "ret_1h_atr", "ret_4h_atr", "ret_12h_atr", "ret_24h_atr",
    "vol_zscore_72", "rsi_norm", "bollinger_pct_b",
    "vol_compression", "bar_pressure", "trend_spread_atr",
    "relative_strength_24h"
]

TIMEOUT_BARS = 24
SL_MULT = 0.90
TP_MULT = 2.20
FEE_MAKER_PCT = 0.0002
FEE_TAKER_PCT = 0.0005
INITIAL_CAPITAL = 200.0
RISK_PCT = 0.015
COOLDOWN_HOURS = 12

unique_ts = np.sort(df["timestamp"].unique())
total_bars = len(unique_ts)

TRAIN_BARS = 4380  # 6 месяцев
TEST_BARS = 1460   # 2 месяца
PURGE_BARS = 24

folds = []
start_idx = 0
fold_num = 1

while start_idx + TRAIN_BARS + PURGE_BARS + TEST_BARS <= total_bars:
    train_start = unique_ts[start_idx]
    train_end = unique_ts[start_idx + TRAIN_BARS]
    test_start = unique_ts[start_idx + TRAIN_BARS + PURGE_BARS]
    test_end = unique_ts[min(start_idx + TRAIN_BARS + PURGE_BARS + TEST_BARS, total_bars - 1)]

    folds.append({
        "fold": fold_num,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end
    })
    start_idx += TEST_BARS
    fold_num += 1

print("=" * 70)
print(f"  PURGED WALK-FORWARD ТЕСТИРОВАНИЕ: {len(folds)} ФОЛДОВ (18 МЕСЯЦЕВ OOS)")
print("=" * 70)

equity = INITIAL_CAPITAL
all_trades = []
full_oos_equity = [equity]
fold_metrics = []

for f in folds:
    f_num = f["fold"]
    train_mask = (df["timestamp"] >= f["train_start"]) & (df["timestamp"] <= f["train_end"]) & (df["is_setup"] == 1) & (~df["meta_label"].isna())
    test_mask = (df["timestamp"] >= f["test_start"]) & (df["timestamp"] <= f["test_end"])

    X_train = df.loc[train_mask, FEATURE_COLS].to_numpy(dtype=np.float32)
    y_train = df.loc[train_mask, "meta_label"].to_numpy(dtype=np.int32)

    if len(X_train) < 100 or sum(y_train) < 10:
        continue

    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        tree_method="hist"
    )
    model.fit(X_train, y_train)

    f_test_df = df[test_mask].copy().reset_index(drop=True)
    f_test_X = f_test_df[FEATURE_COLS].to_numpy(dtype=np.float32)
    f_test_df["pred_prob"] = model.predict_proba(f_test_X)[:, 1]
    f_test_df["ranking_score"] = 0.40 * f_test_df["pred_prob"] + 0.60 * f_test_df["relative_strength_24h"]

    fold_test_ts = np.sort(f_test_df["timestamp"].unique())
    f_grouped = {ts: g for ts, g in f_test_df.groupby("timestamp")}

    fold_start_equity = equity
    fold_trades = []
    open_positions = []
    pending_orders = []
    asset_cooldown = {}

    for cur_ts in fold_test_ts:
        if cur_ts not in f_grouped:
            full_oos_equity.append(equity)
            continue

        bar_df = f_grouped[cur_ts]
        cur_datetime = bar_df["datetime"].iloc[0]

        # 1. Проверка лимитных ордеров
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

            if cur_l <= limit_p <= cur_h and len(open_positions) == 0:
                atr = order["atr"]
                sl_price = limit_p - (atr * SL_MULT)
                tp_price = limit_p + (atr * TP_MULT)

                risk_usd = equity * RISK_PCT
                sl_dist_pct = (limit_p - sl_price) / limit_p
                pos_usd = max(risk_usd / sl_dist_pct, 12.0)

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

        # 2. Сопровождение позиций
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
                trade_record = {
                    "fold": f_num,
                    "coin": coin,
                    "entry_time": pos["entry_time"],
                    "exit_time": cur_datetime,
                    "entry": pos["entry_price"],
                    "exit": exit_p,
                    "pnl": pnl_usd,
                    "reason": reason
                }
                fold_trades.append(trade_record)
                all_trades.append(trade_record)
                closed_positions.append(pos)

        for pos in closed_positions:
            open_positions.remove(pos)

        full_oos_equity.append(equity)

        # 3. Выставление новых лимитов
        if len(open_positions) == 0 and len(pending_orders) == 0:
            candidates = bar_df[
                (bar_df["master_gate_open"] == 1) &
                (bar_df["btc_log_ret_24h"] > 0) &
                (bar_df["is_setup"] == 1) &
                (bar_df["pred_prob"] >= 0.55) &
                (bar_df["relative_strength_24h"] > 0.01)
            ]

            valid_candidates = [
                row_c for _, row_c in candidates.iterrows()
                if asset_cooldown.get(row_c["coin"], 0) <= cur_ts
            ]

            if valid_candidates:
                df_valid = pd.DataFrame(valid_candidates)
                best = df_valid.sort_values("ranking_score", ascending=False).iloc[0]
                pending_orders.append({
                    "coin": best["coin"],
                    "limit_price": best["ema_20"],
                    "atr": best["atr_14"],
                    "ttl": 2
                })

    # Метрики фолда
    df_f_trades = pd.DataFrame(fold_trades)
    f_wins = df_f_trades[df_f_trades["pnl"] > 0] if not df_f_trades.empty else pd.DataFrame()
    f_losses = df_f_trades[df_f_trades["pnl"] < 0] if not df_f_trades.empty else pd.DataFrame()
    f_wr = (len(f_wins) / len(df_f_trades)) * 100.0 if not df_f_trades.empty else 0.0
    f_pf = (f_wins["pnl"].sum() / abs(f_losses["pnl"].sum())) if not f_losses.empty else 0.0
    f_pnl_pct = ((equity - fold_start_equity) / fold_start_equity) * 100.0

    t_start_dt = pd.to_datetime(f["test_start"], unit="ms").strftime("%Y-%m-%d")
    t_end_dt = pd.to_datetime(f["test_end"], unit="ms").strftime("%Y-%m-%d")

    print(f"Фолд {f_num} [{t_start_dt} -> {t_end_dt}] | Сделок: {len(df_f_trades):<2} | WR: {f_wr:4.1f}% | PF: {f_pf:5.2f} | PnL: {f_pnl_pct:+6.2f}% | Баланс: ${equity:,.2f}")
    fold_metrics.append({
        "fold": f_num, "period": f"{t_start_dt} -> {t_end_dt}",
        "trades": len(df_f_trades), "wr": f_wr, "pf": f_pf, "pnl_pct": f_pnl_pct
    })

df_all_trades = pd.DataFrame(all_trades)
eq_arr = np.array(full_oos_equity)
peaks = np.maximum.accumulate(eq_arr)
dds = (peaks - eq_arr) / peaks
max_dd = dds.max() * 100.0
total_pnl = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

all_wins = df_all_trades[df_all_trades["pnl"] > 0] if not df_all_trades.empty else pd.DataFrame()
all_losses = df_all_trades[df_all_trades["pnl"] < 0] if not df_all_trades.empty else pd.DataFrame()
total_wr = (len(all_wins) / len(df_all_trades)) * 100.0 if not df_all_trades.empty else 0.0
total_pf = (all_wins["pnl"].sum() / abs(all_losses["pnl"].sum())) if not all_losses.empty else 0.0

print("=" * 70)
print("  ИТОГОВЫЙ ОТЧЕТ СКВОЗНОЙ WALK-FORWARD ВАЛИДАЦИИ (18 МЕСЯЦЕВ)")
print("=" * 70)
print(f"Стартовый капитал      : ${INITIAL_CAPITAL:,.2f}")
print(f"Конечный капитал       : ${equity:,.2f}")
print(f"Чистая доходность      : {total_pnl:+.2f}%")
print(f"Максимальная просадка  : -{max_dd:.2f}%")
print(f"Всего сделок (OOS)     : {len(df_all_trades)} (~{len(df_all_trades)/18.0:.1f} сделок/мес)")
print(f"Итоговый Win Rate      : {total_wr:.1f}%")
print(f"Итоговый Profit Factor : {total_pf:.3f}")

if not df_all_trades.empty:
    print("\nРаспределение сделок по монетам:")
    print(df_all_trades["coin"].value_counts().to_string())
    print("\nИсходы сделок:")
    print(df_all_trades["reason"].value_counts().to_string())

# График эквити OOS
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
df_all_trades.to_csv(results_dir / "walkforward_trades.csv", index=False)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(eq_arr, label="Walk-Forward Out-Of-Sample Equity ($)", color="#1f77b4", lw=1.5)
plt.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.7)
plt.title("Purged Walk-Forward Continuous Equity (Jul 2023 - Dec 2024)")
plt.ylabel("Portfolio ($)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(-dds * 100.0, label="Drawdown (%)", color="#d62728", lw=1.2)
plt.ylabel("Drawdown (%)")
plt.xlabel("Hourly Bars")
plt.grid(True, alpha=0.3)
plt.legend()

chart_path = results_dir / "walkforward_chart.png"
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"\n[+] График сквозного теста сохранен в: {chart_path}")
