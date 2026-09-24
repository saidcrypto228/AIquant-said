#!/usr/bin/env python3
"""
Final 4H Trend-Following Engine (2023-2024).
- Universe: SOL, NEAR.
- No BE truncation (Let winners run).
- 1 asset = max 1 position.
- Trailing exit strictly on 4H candle close below EMA20.
- Noise-proof stop: 1.8 ATR.
- Risk: 1.5% equity.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data/history_deep")
TARGET_COINS = ["SOL", "NEAR"]

INITIAL_CAPITAL = 200.0
RISK_PCT = 0.015
SL_MULT = 1.80
FEE_MAKER_PCT = 0.0002
FEE_TAKER_PCT = 0.0005

def load_and_resample_4h(coin: str) -> pd.DataFrame:
    fpath = DATA_DIR / f"{coin}_1h.csv"
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_csv(fpath)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")

    ohlc = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "timestamp": "first"
    }
    df_4h = df.resample("4h").agg(ohlc).dropna().reset_index()

    c = df_4h["close"]
    h = df_4h["high"]
    l = df_4h["low"]

    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df_4h["atr_14"] = tr.rolling(14, min_periods=14).mean()
    df_4h["ema_20"] = c.ewm(span=20, adjust=False).mean()
    df_4h["ema_50"] = c.ewm(span=50, adjust=False).mean()
    df_4h["coin"] = coin
    df_4h["log_ret_72h"] = np.log(c / c.shift(18))

    return df_4h.dropna().reset_index(drop=True)

print("=" * 65)
print("  ФИНАЛЬНЫЙ ТЕСТ: ЧИСТЫЙ СВИНГ-ТРЕЙЛИНГ SOL & NEAR")
print("=" * 65)

btc_4h = load_and_resample_4h("BTC")
btc_4h["btc_bull"] = (btc_4h["close"] > btc_4h["ema_50"]) & (btc_4h["ema_20"] > btc_4h["ema_50"])
btc_map = btc_4h.set_index("datetime")[["btc_bull", "log_ret_72h"]].to_dict(orient="index")

coin_dfs = []
for coin in TARGET_COINS:
    c_df = load_and_resample_4h(coin)
    c_df["btc_bull"] = c_df["datetime"].map(lambda dt: btc_map.get(dt, {}).get("btc_bull", False))
    c_df["btc_ret_72h"] = c_df["datetime"].map(lambda dt: btc_map.get(dt, {}).get("log_ret_72h", 0.0))
    c_df["relative_strength"] = c_df["log_ret_72h"] - c_df["btc_ret_72h"]
    coin_dfs.append(c_df)

universe_4h = pd.concat(coin_dfs, ignore_index=True)
universe_4h = universe_4h.sort_values(["datetime", "coin"]).reset_index(drop=True)

unique_datetimes = np.sort(universe_4h["datetime"].unique())
grouped = {dt: g for dt, g in universe_4h.groupby("datetime")}

equity = INITIAL_CAPITAL
equity_curve = [equity]
trades = []
open_positions = []
pending_orders = []

for dt_idx, cur_dt in enumerate(unique_datetimes):
    if cur_dt not in grouped:
        equity_curve.append(equity)
        continue

    bar_df = grouped[cur_dt]

    # 1. Лимитные ордера
    still_pending = []
    for order in pending_orders:
        coin = order["coin"]
        if any(p["coin"] == coin for p in open_positions):
            continue

        row = bar_df[bar_df["coin"] == coin]
        if row.empty:
            continue

        cur_l = row["low"].values[0]
        cur_h = row["high"].values[0]
        limit_p = order["limit_price"]

        if cur_l <= limit_p <= cur_h and len(open_positions) < 2:
            atr = order["atr"]
            sl_price = limit_p - (atr * SL_MULT)
            risk_usd = equity * RISK_PCT
            sl_dist_pct = (limit_p - sl_price) / limit_p
            pos_usd = max(risk_usd / sl_dist_pct, 15.0)

            open_positions.append({
                "coin": coin,
                "entry_time": cur_dt,
                "entry_price": limit_p,
                "pos_usd": pos_usd,
                "sl_price": sl_price,
                "dur": 0
            })
        else:
            order["ttl"] -= 1
            if order["ttl"] > 0:
                still_pending.append(order)
    pending_orders = still_pending

    # 2. Сопровождение позиций: Трейлинг EMA20
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
        cur_ema20 = row["ema_20"].values[0]
        pos["dur"] += 1

        exit_p = None
        reason = ""

        if cur_l <= pos["sl_price"]:
            exit_p = pos["sl_price"]
            reason = "SL_WIDE"
        elif pos["dur"] >= 2 and cur_c < cur_ema20:
            exit_p = cur_c
            reason = "TRAILING_EMA20"

        if exit_p is not None:
            raw_ret = (exit_p / pos["entry_price"]) - 1.0
            fees = FEE_MAKER_PCT + FEE_TAKER_PCT
            pnl_usd = pos["pos_usd"] * (raw_ret - fees)

            equity += pnl_usd
            trades.append({
                "coin": coin,
                "entry_time": pos["entry_time"],
                "exit_time": cur_dt,
                "entry": pos["entry_price"],
                "exit": exit_p,
                "pnl": pnl_usd,
                "dur_4h_bars": pos["dur"],
                "ret_pct": raw_ret * 100.0,
                "reason": reason
            })
            closed_positions.append(pos)

    for pos in closed_positions:
        open_positions.remove(pos)

    equity_curve.append(equity)

    # 3. Выбор новых ордеров (1 актив = максимум 1 позиция)
    active_coins = {p["coin"] for p in open_positions} | {o["coin"] for o in pending_orders}

    if len(open_positions) < 2 and len(pending_orders) < (2 - len(open_positions)):
        candidates = bar_df[
            (bar_df["btc_bull"] == True) &
            (bar_df["close"] > bar_df["ema_50"]) &
            (bar_df["ema_20"] > bar_df["ema_50"]) &
            (bar_df["relative_strength"] > 0.0) &
            (~bar_df["coin"].isin(active_coins))
        ]

        if not candidates.empty:
            for _, row_c in candidates.sort_values("relative_strength", ascending=False).iterrows():
                if len(open_positions) + len(pending_orders) < 2:
                    pending_orders.append({
                        "coin": row_c["coin"],
                        "limit_price": row_c["ema_20"],
                        "atr": row_c["atr_14"],
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

print("=" * 65)
print("  РЕЗУЛЬТАТЫ СВИНГ-МОДЕЛЕЙ (БЕЗ УДУШЕНИЯ БЕЗУБЫТКОМ)")
print("=" * 65)
print(f"Стартовый депозит     : ${INITIAL_CAPITAL:,.2f}")
print(f"Конечный баланс       : ${equity:,.2f}")
print(f"Чистый PnL            : {total_pnl:+.2f}%")
print(f"Максимальная просадка : -{max_dd:.2f}%")
print(f"Всего сделок          : {len(df_trades)} (~{len(df_trades)/24.0:.1f} сделок/мес)")
print(f"Win Rate              : {wr:.1f}%")
print(f"Profit Factor         : {pf:.3f}")

if not df_trades.empty:
    avg_win = win_trades["pnl"].mean()
    avg_loss = abs(loss_trades["pnl"].mean()) if not loss_trades.empty else 1.0
    print(f"Средний выигрыш       : +${avg_win:.2f}")
    print(f"Средний проигрыш      : -${avg_loss:.2f}")
    print(f"Фактический R:R       : {avg_win / avg_loss:.2f} : 1")
    print("\nИсходы сделок:")
    print(df_trades["reason"].value_counts().to_string())
    print("\nПоказатели по монетам:")
    for coin, g in df_trades.groupby("coin"):
        c_win = g[g["pnl"] > 0]
        c_loss = g[g["pnl"] < 0]
        c_pf = c_win["pnl"].sum() / abs(c_loss["pnl"].sum()) if not c_loss.empty else 0
        print(f"  {coin:<5}: {len(g)} сделок | WR: {len(c_win)/len(g)*100:.1f}% | PF: {c_pf:.2f} | PnL: +${g['pnl'].sum():.2f}")

print("=" * 65)
