#!/usr/bin/env python3
"""
4H Swing Trend-Following Engine (2023-2024).
Resamples 1h data to 4h candles.
Limit entry at 4H EMA20, Wide Noise-Proof Stop (1.8 ATR),
Trailing exit on 4H candle close below EMA20.
Sizing: 1.5% equity risk ($3 on $200).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path("data/history_deep")
COINS = ["BTC", "SOL", "NEAR", "SUI", "ETH"]

INITIAL_CAPITAL = 200.0
RISK_PCT = 0.015
SL_MULT = 1.80          # Шумогасящий стоп
FEE_MAKER_PCT = 0.0002  # Лимитный вход
FEE_TAKER_PCT = 0.0005  # Выход по трейлингу / стопу

def load_and_resample_4h(coin: str) -> pd.DataFrame:
    fpath = DATA_DIR / f"{coin}_1h.csv"
    if not fpath.exists():
        return pd.DataFrame()
    df = pd.read_csv(fpath)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")

    # Агрегация часовиков в 4-часовые бары
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "timestamp": "first"
    }
    df_4h = df.resample("4h").agg(ohlc).dropna().reset_index()

    # Индикаторы 4H
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
    df_4h["log_ret_72h"] = np.log(c / c.shift(18)) # 18 баров по 4h = 72h

    return df_4h.dropna().reset_index(drop=True)

print("=" * 65)
print("  РЕСЕМПЛИНГ 1H -> 4H И СБОРКА ТРЕНДОВОГО СВИНГ-ПУЛА")
print("=" * 65)

btc_4h = load_and_resample_4h("BTC")
if btc_4h.empty:
    print("[-] Не найдены данные BTC!")
    sys.exit(1)

btc_4h["btc_bull"] = (btc_4h["close"] > btc_4h["ema_50"]) & (btc_4h["ema_20"] > btc_4h["ema_50"])
btc_map = btc_4h.set_index("datetime")[["btc_bull", "log_ret_72h"]].to_dict(orient="index")

coin_dfs = []
for coin in ["SOL", "NEAR", "SUI", "ETH"]:
    c_df = load_and_resample_4h(coin)
    if c_df.empty:
        continue
    c_df["btc_bull"] = c_df["datetime"].map(lambda dt: btc_map.get(dt, {}).get("btc_bull", False))
    c_df["btc_ret_72h"] = c_df["datetime"].map(lambda dt: btc_map.get(dt, {}).get("log_ret_72h", 0.0))
    c_df["relative_strength"] = c_df["log_ret_72h"] - c_df["btc_ret_72h"]
    coin_dfs.append(c_df)
    print(f"[+] {coin:<5}: сформировано {len(c_df):,} 4-часовых свечей")

universe_4h = pd.concat(coin_dfs, ignore_index=True)
universe_4h = universe_4h.sort_values(["datetime", "coin"]).reset_index(drop=True)

unique_datetimes = np.sort(universe_4h["datetime"].unique())
grouped = {dt: g for dt, g in universe_4h.groupby("datetime")}

equity = INITIAL_CAPITAL
equity_curve = [equity]
trades = []
open_positions = []
pending_orders = []

print(f"\n[*] Запуск симуляции 4H Свинг-трейдинга ({len(unique_datetimes):,} свечей)...")

for dt_idx, cur_dt in enumerate(unique_datetimes):
    if cur_dt not in grouped:
        equity_curve.append(equity)
        continue

    bar_df = grouped[cur_dt]

    # 1. Проверка лимитных ордеров (Maker вход на EMA20)
    still_pending = []
    for order in pending_orders:
        coin = order["coin"]
        row = bar_df[bar_df["coin"] == coin]
        if row.empty:
            continue

        cur_l = row["low"].values[0]
        cur_h = row["high"].values[0]
        limit_p = order["limit_price"]

        # Если 4H бар задел уровень EMA20
        if cur_l <= limit_p <= cur_h and len(open_positions) < 2:
            atr = order["atr"]
            sl_price = limit_p - (atr * SL_MULT)
            risk_usd = equity * RISK_PCT
            sl_dist_pct = (limit_p - sl_price) / limit_p
            pos_usd = max(risk_usd / sl_dist_pct, 15.0) # Не менее $15

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

    # 2. Сопровождение открытых позиций (Широкий стоп + Трейлинг EMA20)
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

        # Полный стоп (1.8 ATR)
        if cur_l <= pos["sl_price"]:
            exit_p = pos["sl_price"]
            reason = "SL_WIDE"
        # Трейлинг-выход: 4H свеча закрылась НИЖЕ EMA20 (тренд выдохся)
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

    # 3. Выбор новых кандидатов
    if len(open_positions) < 2 and len(pending_orders) == 0:
        candidates = bar_df[
            (bar_df["btc_bull"] == True) &
            (bar_df["close"] > bar_df["ema_50"]) &
            (bar_df["ema_20"] > bar_df["ema_50"]) &
            (bar_df["relative_strength"] > 0.0)
        ]

        if not candidates.empty:
            # Сортируем по относительной силе к BTC
            best = candidates.sort_values("relative_strength", ascending=False).iloc[0]
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

print("=" * 65)
print("  РЕЗУЛЬТАТЫ 4H SWING TREND СИМУЛЯЦИИ (2023-2024)")
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
    print("\nРаспределение по монетам:")
    print(df_trades["coin"].value_counts().to_string())
    print("\nИсходы сделок:")
    print(df_trades["reason"].value_counts().to_string())
    print("\nТоп-5 лучших трейдов:")
    print(df_trades.sort_values("pnl", ascending=False)[["coin", "entry", "exit", "ret_pct", "pnl"]].head().to_string())

# График
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
df_trades.to_csv(results_dir / "swing_4h_trades.csv", index=False)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(eq_arr, label="4H Swing Portfolio ($)", color="#2ca02c", lw=1.5)
plt.axhline(INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.7)
plt.title("4H Swing Trend-Following Performance (2023-2024)")
plt.ylabel("Balance ($)")
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(-dds * 100.0, label="Drawdown (%)", color="#d62728", lw=1.2)
plt.ylabel("Drawdown (%)")
plt.xlabel("4H Bars")
plt.grid(True, alpha=0.3)
plt.legend()

chart_path = results_dir / "swing_4h_chart.png"
plt.tight_layout()
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"\n[+] График сохранен в: {chart_path}")
print("=" * 65)
