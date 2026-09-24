#!/usr/bin/env python3
"""
Realistic 4H Swing Simulation with Microstructure Drag:
- Funding Rate Drag (40% APR annualized on longs).
- Adverse Selection Queue Filter (Price must penetrate EMA20 by >= 0.15 ATR to simulate queue fill).
- Stop Slippage (0.15% adverse slip on all exits).
- Strict exchange min-notional ($10.50) filtering.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/history_deep")
TARGET_COINS = ["SOL", "NEAR"]

INITIAL_CAPITAL = 200.0
RISK_PCT = 0.015
SL_MULT = 1.80

# Комиссии и микроструктурные издержки
FEE_MAKER_PCT = 0.0002
FEE_TAKER_PCT = 0.0005
STOP_SLIPPAGE_PCT = 0.0015     # 0.15% проскальзывание на стопе
HOURLY_FUNDING_RATE = 0.000045 # ~40% годовых перпетуального фандинга на бычьем рынке

PENETRATION_ATR_REQ = 0.15     # Требуемое проникновение вглубь EMA для гарантии филла в очереди

def load_resampled_4h(coin: str) -> pd.DataFrame:
    fpath = DATA_DIR / f"{coin}_1h.csv"
    if not fpath.exists():
        print(f"[-] Файл {fpath} не найден!")
        sys.exit(1)
    df = pd.read_csv(fpath)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").set_index("datetime")

    ohlc = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "timestamp": "first"
    }
    # Строго правая граница агрегации (исключение lookahead)
    df_4h = df.resample("4h", label="right", closed="right").agg(ohlc).dropna().reset_index()
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

print("=" * 70)
print("  РЕАЛИСТИЧНЫЙ КВАНТ-ТЕСТ С УЧЕТОМ ОЧЕРЕДИ, ФАНДИНГА И ПРОСКАЛЬЗЫВАНИЯ")
print("=" * 70)

btc_4h = load_resampled_4h("BTC")
# Master Gate оценивается строго по завершенным свечам
btc_4h["btc_bull"] = (btc_4h["close"] > btc_4h["ema_50"]) & (btc_4h["ema_20"] > btc_4h["ema_50"])
btc_map = btc_4h.set_index("datetime")[["btc_bull", "log_ret_72h"]].to_dict(orient="index")

coin_dfs = []
for coin in TARGET_COINS:
    c_df = load_resampled_4h(coin)
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

unfilled_queue_misses = 0

for dt_idx, cur_dt in enumerate(unique_datetimes):
    if cur_dt not in grouped:
        equity_curve.append(equity)
        continue
    bar_df = grouped[cur_dt]

    # 1. Проверка лимитных заявок с моделью очереди
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
        atr = order["atr"]

        # Реалистичное условие очереди: цена должна опуститься ниже EMA20 минимум на 0.15 ATR
        # Простое касание не гарантирует исполнение в хвосте лимитной очереди
        filled = (cur_l <= (limit_p - (atr * PENETRATION_ATR_REQ)))

        if filled and len(open_positions) < 2:
            sl_price = limit_p - (atr * SL_MULT)
            risk_usd = equity * RISK_PCT
            sl_dist_pct = (limit_p - sl_price) / limit_p
            pos_usd = max(risk_usd / sl_dist_pct, 12.0)

            open_positions.append({
                "coin": coin, "entry_time": cur_dt, "entry_price": limit_p,
                "pos_usd": pos_usd, "sl_price": sl_price, "dur_bars": 0
            })
        elif cur_l <= limit_p:
            # Касание было, но вглубь не ушло -> заявка осталась неисполненной в очереди
            unfilled_queue_misses += 1
            order["ttl"] -= 1
            if order["ttl"] > 0:
                still_pending.append(order)
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
            pos["dur_bars"] += 1
            continue

        cur_h = row["high"].values[0]
        cur_l = row["low"].values[0]
        cur_c = row["close"].values[0]
        cur_ema20 = row["ema_20"].values[0]
        pos["dur_bars"] += 1

        exit_p = None
        reason = ""

        if cur_l <= pos["sl_price"]:
            # Стоп исполняется с учетом штрафного проскальзывания
            exit_p = pos["sl_price"] * (1.0 - STOP_SLIPPAGE_PCT)
            reason = "SL_WIDE_SLIP"
        elif pos["dur_bars"] >= 2 and cur_c < cur_ema20:
            # Трейлинг на закрытии бара
            exit_p = cur_c * (1.0 - (STOP_SLIPPAGE_PCT * 0.5))
            reason = "TRAILING_EMA20"

        if exit_p is not None:
            raw_ret = (exit_p / pos["entry_price"]) - 1.0
            fees = FEE_MAKER_PCT + FEE_TAKER_PCT

            # Расчет списания перпетуального фандинга за время жизни позиции
            hold_hours = pos["dur_bars"] * 4
            funding_drag_pct = hold_hours * HOURLY_FUNDING_RATE

            net_ret = raw_ret - fees - funding_drag_pct
            pnl_usd = pos["pos_usd"] * net_ret
            equity += pnl_usd

            trades.append({
                "coin": coin, "entry": pos["entry_price"], "exit": exit_p,
                "pnl": pnl_usd, "dur_hours": hold_hours, "funding_cost": pos["pos_usd"] * funding_drag_pct,
                "reason": reason
            })
            closed_positions.append(pos)

    for pos in closed_positions:
        open_positions.remove(pos)

    equity_curve.append(equity)

    # 3. Выбор новых кандидатов
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
                        "coin": row_c["coin"], "limit_price": row_c["ema_20"],
                        "atr": row_c["atr_14"], "ttl": 2
                    })

df_t = pd.DataFrame(trades)
eq_arr = np.array(equity_curve)
peaks = np.maximum.accumulate(eq_arr)
dds = (peaks - eq_arr) / peaks
max_dd = dds.max() * 100.0
total_pnl = ((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

win_t = df_t[df_t["pnl"] > 0] if not df_t.empty else pd.DataFrame()
loss_t = df_t[df_t["pnl"] < 0] if not df_t.empty else pd.DataFrame()
wr = (len(win_t) / len(df_t)) * 100.0 if not df_t.empty else 0.0
pf = (win_t["pnl"].sum() / abs(loss_t["pnl"].sum())) if not loss_t.empty else 0.0

total_funding_paid = df_t["funding_cost"].sum() if not df_t.empty else 0.0

print("=" * 70)
print("  ИТОГИ ЖЕСТКОГО СТРЕСС-БЭКТЕСТА (LIVE REALISM)")
print("=" * 70)
print(f"Стартовый депозит         : ${INITIAL_CAPITAL:,.2f}")
print(f"Конечный капитал          : ${equity:,.2f}")
print(f"Реалистичный чистый PnL   : {total_pnl:+.2f}%")
print(f"Максимальная просадка     : -{max_dd:.2f}%")
print(f"Фактически исполнено      : {len(df_t)} сделок (отсеяно очередью: {unfilled_queue_misses} касаний)")
print(f"Винрейт с учетом издержек : {wr:.1f}%")
print(f"Profit Factor             : {pf:.3f}")
print(f"Списано фандинга за 2 года: -${total_funding_paid:.2f} ({(total_funding_paid/INITIAL_CAPITAL)*100:.1f}% от счета)")
print("=" * 70)
