#!/usr/bin/env python3
"""
4H Swing Engine with 1H Confirmation (Anti-Adverse Selection).
Rules:
1. 4H Trend up (EMA20 > EMA50, BTC Gate open, RS > 0).
2. Price enters 4H EMA20 zone.
3. TRIGGER: Enter only after 1H candle closes GREEN (Close > Open) near EMA20.
4. Frictions included: Funding (40% APR), Stop slippage (0.15%), Taker exit fees.
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

FEE_MAKER_PCT = 0.0002
FEE_TAKER_PCT = 0.0005
STOP_SLIPPAGE_PCT = 0.0015
HOURLY_FUNDING_RATE = 0.000045 # ~40% APR

def load_data(coin: str) -> pd.DataFrame:
    fpath = DATA_DIR / f"{coin}_1h.csv"
    if not fpath.exists():
        sys.exit(1)
    df = pd.read_csv(fpath)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

print("=" * 70)
print("  ТЕСТ: 4H СВИНГ С 1H ПОДТВЕРЖДЕНИЕМ (УСТРАНЕНИЕ ADVERSE SELECTION)")
print("=" * 70)

# 1. Загрузка часовиков
btc_raw = load_data("BTC")
sol_raw = load_data("SOL")
near_raw = load_data("NEAR")

# Расчет 4H индикаторов через корректный resample
def add_4h_indicators(df_1h: pd.DataFrame, coin: str) -> pd.DataFrame:
    df = df_1h.copy().set_index("datetime")
    ohlc = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "timestamp": "first"}
    df_4h = df.resample("4h", label="right", closed="right").agg(ohlc).dropna()

    c = df_4h["close"]
    h = df_4h["high"]
    l = df_4h["low"]
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df_4h["atr_4h"] = tr.rolling(14, min_periods=14).mean()
    df_4h["ema20_4h"] = c.ewm(span=20, adjust=False).mean()
    df_4h["ema50_4h"] = c.ewm(span=50, adjust=False).mean()
    df_4h["ret_72h"] = np.log(c / c.shift(18))

    # Возвращаем на часовой таймфрейм со сдвигом на 1 бар 4H (исключение lookahead)
    df_4h_shifted = df_4h[["atr_4h", "ema20_4h", "ema50_4h", "ret_72h"]].shift(1)

    merged = pd.merge_asof(
        df_1h.sort_values("datetime"),
        df_4h_shifted.reset_index().sort_values("datetime"),
        on="datetime",
        direction="backward"
    )
    merged["coin"] = coin
    return merged.dropna().reset_index(drop=True)

btc_df = add_4h_indicators(btc_raw, "BTC")
btc_df["btc_bull"] = (btc_df["close"] > btc_df["ema50_4h"]) & (btc_df["ema20_4h"] > btc_df["ema50_4h"])
btc_map = btc_df.set_index("datetime")[["btc_bull", "ret_72h"]].to_dict(orient="index")

coin_dfs = []
for raw, coin in [(sol_raw, "SOL"), (near_raw, "NEAR")]:
    c_df = add_4h_indicators(raw, coin)
    c_df["btc_bull"] = c_df["datetime"].map(lambda dt: btc_map.get(dt, {}).get("btc_bull", False))
    c_df["btc_ret"] = c_df["datetime"].map(lambda dt: btc_map.get(dt, {}).get("ret_72h", 0.0))
    c_df["rs"] = c_df["ret_72h"] - c_df["btc_ret"]
    coin_dfs.append(c_df)

df_all = pd.concat(coin_dfs, ignore_index=True).sort_values(["datetime", "coin"]).reset_index(drop=True)
unique_dts = np.sort(df_all["datetime"].unique())
grouped = {dt: g for dt, g in df_all.groupby("datetime")}

equity = INITIAL_CAPITAL
equity_curve = [equity]
trades = []
open_positions = []

for cur_dt in unique_dts:
    if cur_dt not in grouped:
        equity_curve.append(equity)
        continue
    bar_df = grouped[cur_dt]

    # 1. Сопровождение позиций
    closed_pos = []
    for pos in open_positions:
        coin = pos["coin"]
        row = bar_df[bar_df["coin"] == coin]
        if row.empty:
            pos["dur_hours"] += 1
            continue

        cur_h = row["high"].values[0]
        cur_l = row["low"].values[0]
        cur_c = row["close"].values[0]
        cur_ema20 = row["ema20_4h"].values[0]
        pos["dur_hours"] += 1

        exit_p = None
        reason = ""

        if cur_l <= pos["sl_price"]:
            exit_p = pos["sl_price"] * (1.0 - STOP_SLIPPAGE_PCT)
            reason = "SL_WIDE"
        elif pos["dur_hours"] >= 8 and cur_c < cur_ema20:
            exit_p = cur_c * (1.0 - (STOP_SLIPPAGE_PCT * 0.5))
            reason = "TRAILING_EMA20"

        if exit_p is not None:
            raw_ret = (exit_p / pos["entry_price"]) - 1.0
            fees = FEE_MAKER_PCT + FEE_TAKER_PCT
            funding_drag = pos["dur_hours"] * HOURLY_FUNDING_RATE
            pnl_usd = pos["pos_usd"] * (raw_ret - fees - funding_drag)

            equity += pnl_usd
            trades.append({
                "coin": coin, "entry": pos["entry_price"], "exit": exit_p,
                "pnl": pnl_usd, "dur_hours": pos["dur_hours"], "funding": pos["pos_usd"] * funding_drag,
                "reason": reason
            })
            closed_pos.append(pos)

    for p in closed_pos:
        open_positions.remove(p)

    equity_curve.append(equity)

    # 2. Вход по подтверждению (1H Reversal near 4H EMA20)
    active_coins = {p["coin"] for p in open_positions}
    if len(open_positions) < 2:
        for _, row in bar_df.iterrows():
            coin = row["coin"]
            if coin in active_coins or len(open_positions) >= 2:
                continue

            # Условия:
            # 1. 4H тренд вверх, BTC бычий, RS > 0
            # 2. Цена находится в зоне отката к 4H EMA20 (не выше +1% от средней)
            # 3. ПОДТВЕРЖДЕНИЕ: часовая свеча зеленая (close > open) и коснулась EMA20
            is_trend = row["btc_bull"] and (row["close"] > row["ema50_4h"]) and (row["ema20_4h"] > row["ema50_4h"]) and (row["rs"] > 0)
            in_pullback_zone = (row["low"] <= row["ema20_4h"] * 1.005) and (row["close"] >= row["ema20_4h"] * 0.985)
            is_green_reversal = (row["close"] > row["open"]) # Покупатели защитили уровень

            if is_trend and in_pullback_zone and is_green_reversal:
                entry_p = row["close"]
                atr = row["atr_4h"]
                sl_p = entry_p - (atr * SL_MULT)

                risk_usd = equity * RISK_PCT
                sl_dist_pct = (entry_p - sl_p) / entry_p
                pos_usd = max(risk_usd / sl_dist_pct, 12.0)

                open_positions.append({
                    "coin": coin, "entry_time": cur_dt, "entry_price": entry_p,
                    "pos_usd": pos_usd, "sl_price": sl_p, "dur_hours": 0
                })
                active_coins.add(coin)

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

print("=" * 70)
print("  ИТОГИ СИМУЛЯЦИИ С 1H ПОДТВЕРЖДЕНИЕМ (CONFIRMED PULLBACK)")
print("=" * 70)
print(f"Стартовый депозит         : ${INITIAL_CAPITAL:,.2f}")
print(f"Конечный капитал          : ${equity:,.2f}")
print(f"Чистый PnL с издержками   : {total_pnl:+.2f}%")
print(f"Максимальная просадка     : -{max_dd:.2f}%")
print(f"Всего сделок              : {len(df_t)}")
print(f"Win Rate                  : {wr:.1f}%")
print(f"Profit Factor             : {pf:.3f}")
if not df_t.empty:
    print(f"Списано фандинга за 2 года: -${df_t['funding'].sum():.2f}")
    print(f"Средний выигрыш           : +${win_t['pnl'].mean():.2f}")
    print(f"Средний проигрыш          : -${abs(loss_t['pnl'].mean()):.2f}")
    print(f"Реализованный R:R         : {win_t['pnl'].mean() / abs(loss_t['pnl'].mean()):.2f} : 1")
print("=" * 70)
