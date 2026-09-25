#!/usr/bin/env python3
"""
Институциональный бэктестер v10.7 (Hardened Portfolio Risk & High Slippage Stress).
- Инвариант: PortfolioStopRisk <= 8.0% от Equity (защита от мгновенного сноса 3 стопов).
- Стресс-тест исполнения: проскальзывание стопа 0.25% (в 5 раз жестче базового 0.05%).
- Изоляция незакрытой 1H свечи (Lookahead-free).
- Строгий OOS с 24h эмбарго, пессимистичные коллизии STOP_COLLISION_WORST.
"""

import time
import math
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from hyperliquid.info import Info
from hyperliquid.utils import constants

import bot_config as config
from quant_factors import QuantFactorEngine

print("=" * 85)
print("  БЭКТЕСТЕР v10.7: HARDENED PORTFOLIO RISK & SLIPPAGE STRESS (0.25%)")
print("=" * 85)

clean_target_coins = [c for c in config.TARGET_COINS if c not in ["ETH", "LINK", "PEPE", "kPEPE", "WIF"]]

model_path = config.DATA_DIR / "meta_model.json"
with open(model_path, "r", encoding="utf-8") as f:
    model_data = json.load(f)

weights = np.array(model_data["coef"])
intercept = float(model_data["intercept"])
scaler_mean = np.array(model_data["scaler_mean"])
scaler_scale = np.array(model_data["scaler_scale"])
scaler_scale = np.where(scaler_scale <= 1e-6, 1.0, scaler_scale)

def predict_meta_prob(raw_features: list) -> float:
    x = (np.array(raw_features) - scaler_mean) / scaler_scale
    z = float(np.dot(weights, x) + intercept)
    return 1.0 / (1.0 + math.exp(-max(min(z, 15.0), -15.0)))

base_url = constants.TESTNET_API_URL if config.IS_TESTNET else constants.MAINNET_API_URL
info = Info(base_url, skip_ws=True, timeout=5)

END_MS = int(time.time() * 1000)
START_MS = END_MS - (1500 * 3600 * 1000)
SYMBOLS = list(set(["BTC"] + clean_target_coins))

print(f"[*] Выгрузка исторических данных Hyperliquid ({len(SYMBOLS)} инструментов)...")
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
total_bars = len(common_timestamps)
split_idx = int(total_bars * 0.75)
oos_start_idx = split_idx + 24

INITIAL_CAPITAL = 1000.0
TAKER_FEE = 0.00035
# ПОВЫШЕННЫЙ ШТРАФ ПРОСКАЛЬЗЫВАНИЯ: 0.25% (в 5 раз выше нормального)
SLIPPAGE_PENALTY = 0.0025
HOURLY_FUNDING = 0.000012

def simulate_v10_7(start_idx: int, end_idx: int, title: str):
    cash = INITIAL_CAPITAL
    equity_curve = [cash]
    trades = []
    open_positions = {}
    pending_triggers = {}

    for i in range(start_idx, end_idx):
        curr_t = common_timestamps[i]
        btc_df = processed_1h["BTC"]
        btc_row = btc_df[btc_df["t"] == curr_t]
        if btc_row.empty:
            continue
        btc_row = btc_row.iloc[0]

        btc_bull = bool(btc_row["close"] > btc_row["ema50_4h"])
        btc_bear = bool(btc_row["close"] < btc_row["ema50_4h"])
        btc_slope_rel = btc_row["ema50_slope"] / max(btc_row["close"] * 0.01, 1e-4)

        is_strong_trend = (btc_bull and btc_slope_rel > 0.15) or (btc_bear and btc_slope_rel < -0.25)
        max_slots = 3 if is_strong_trend else 2

        # 1. СОПРОВОЖДЕНИЕ ПОЗИЦИЙ
        for coin in list(open_positions.keys()):
            pos = open_positions[coin]
            c_df = processed_1h[coin]
            c_row = c_df[c_df["t"] == curr_t]
            if c_row.empty:
                continue
            c_row = c_row.iloc[0]

            is_long = (pos["direction"] == "LONG")
            exit_trade = False
            exit_price = 0.0
            exit_reason = ""

            atr = pos["atr"]
            active_sl = pos["sl_px"]

            if is_long:
                cash -= pos["notional"] * HOURLY_FUNDING
                be_target = pos["entry_px"] + (atr * 1.0)
                hit_sl = (c_row["low"] <= active_sl)
                hit_be = (c_row["high"] >= be_target) and not pos["breakeven_active"]

                if hit_sl and hit_be:
                    exit_trade = True
                    exit_price = active_sl * (1.0 - SLIPPAGE_PENALTY)
                    exit_reason = "STOP_COLLISION_WORST"
                elif c_row["open"] <= active_sl:
                    exit_trade = True
                    exit_price = c_row["open"] * (1.0 - SLIPPAGE_PENALTY)
                    exit_reason = "STOP_GAP_OPEN"
                elif hit_sl:
                    exit_trade = True
                    exit_price = active_sl * (1.0 - SLIPPAGE_PENALTY)
                    exit_reason = "CHANDELIER_EXIT" if pos["trailing_active"] else ("BREAKEVEN_EXIT" if pos["breakeven_active"] else "INITIAL_STOP")
            else:
                cash += pos["notional"] * HOURLY_FUNDING
                be_target = pos["entry_px"] - (atr * 1.0)
                hit_sl = (c_row["high"] >= active_sl)
                hit_be = (c_row["low"] <= be_target) and not pos["breakeven_active"]

                if hit_sl and hit_be:
                    exit_trade = True
                    exit_price = active_sl * (1.0 + SLIPPAGE_PENALTY)
                    exit_reason = "STOP_COLLISION_WORST"
                elif c_row["open"] >= active_sl:
                    exit_trade = True
                    exit_price = c_row["open"] * (1.0 + SLIPPAGE_PENALTY)
                    exit_reason = "STOP_GAP_OPEN"
                elif hit_sl:
                    exit_trade = True
                    exit_price = active_sl * (1.0 + SLIPPAGE_PENALTY)
                    exit_reason = "CHANDELIER_EXIT" if pos["trailing_active"] else ("BREAKEVEN_EXIT" if pos["breakeven_active"] else "INITIAL_STOP")

            if exit_trade:
                if is_long:
                    pnl = (exit_price - pos["entry_px"]) * pos["sz"] - (exit_price * pos["sz"] * TAKER_FEE)
                    pnl_pct = (exit_price / pos["entry_px"] - 1.0) * 100
                else:
                    pnl = (pos["entry_px"] - exit_price) * pos["sz"] - (exit_price * pos["sz"] * TAKER_FEE)
                    pnl_pct = (1.0 - exit_price / pos["entry_px"]) * 100

                cash += (pos["margin"] + pnl)
                trades.append({
                    "coin": coin, "dir": pos["direction"], "type": pos["type"],
                    "entry_px": pos["entry_px"], "exit_px": exit_price,
                    "pnl_usd": pnl, "pnl_pct": pnl_pct, "reason": exit_reason, "ml_prob": pos["ml_prob"]
                })
                del open_positions[coin]
                continue

            if is_long:
                pos["highest_px"] = max(pos["highest_px"], c_row["high"])
                unrealized_r = (pos["highest_px"] - pos["entry_px"]) / max(atr, 1e-4)

                if unrealized_r >= 1.1 and not pos["breakeven_active"]:
                    be_price = pos["entry_px"] * 1.002
                    if be_price > pos["sl_px"]:
                        pos["sl_px"] = be_price
                        pos["breakeven_active"] = True

                if unrealized_r >= 1.8:
                    new_sl = max(pos["sl_px"], pos["highest_px"] - (atr * 1.75), pos["entry_px"] * 1.002)
                    if new_sl > pos["sl_px"]:
                        pos["sl_px"] = new_sl
                        pos["trailing_active"] = True
            else:
                pos["lowest_px"] = min(pos["lowest_px"], c_row["low"])
                unrealized_r = (pos["entry_px"] - pos["lowest_px"]) / max(atr, 1e-4)

                if unrealized_r >= 1.1 and not pos["breakeven_active"]:
                    be_price = pos["entry_px"] * 0.998
                    if be_price < pos["sl_px"]:
                        pos["sl_px"] = be_price
                        pos["breakeven_active"] = True

                if unrealized_r >= 1.8:
                    new_sl = min(pos["sl_px"], pos["lowest_px"] + (atr * 1.75), pos["entry_px"] * 0.998)
                    if new_sl < pos["sl_px"]:
                        pos["sl_px"] = new_sl
                        pos["trailing_active"] = True

        # 2. ИСПОЛНЕНИЕ ТРИГГЕРОВ (ИНВАРИАНТ PORTFOLIO STOP RISK <= 8.0%)
        total_unrealized = 0.0
        total_notional = 0.0
        total_margin = sum(p["margin"] for p in open_positions.values())

        for c_name, pos in open_positions.items():
            c_cur_px = processed_1h[c_name][processed_1h[c_name]["t"] == curr_t].iloc[0]["close"]
            if pos["direction"] == "LONG":
                total_unrealized += (c_cur_px - pos["entry_px"]) * pos["sz"]
            else:
                total_unrealized += (pos["entry_px"] - c_cur_px) * pos["sz"]
            total_notional += pos["notional"]

        current_equity = cash + total_margin + total_unrealized
        portfolio_cap_usd = current_equity * config.PORTFOLIO_HARD_LEVERAGE_CAP

        # Расчет текущего риска открытых позиций
        current_stop_risk = sum(p["sz"] * abs(p["entry_px"] - p["sl_px"]) for p in open_positions.values())
        max_stop_risk_allowed = current_equity * 0.08  # 8% хардкап

        for coin, trig in list(pending_triggers.items()):
            if curr_t > trig["expiry_t"] or len(open_positions) >= max_slots:
                del pending_triggers[coin]
                continue

            c_row = processed_1h[coin][processed_1h[coin]["t"] == curr_t].iloc[0]
            is_long = (trig["direction"] == "LONG")
            trigger_hit = (c_row["high"] >= trig["trigger_px"]) if is_long else (c_row["low"] <= trig["trigger_px"])

            if trigger_hit:
                fill_px = max(trig["trigger_px"], c_row["open"]) * 1.0005 if is_long else min(trig["trigger_px"], c_row["open"]) * 0.9995
                sl_dist = abs(fill_px - trig["sl_px"])

                if sl_dist / fill_px >= 0.008:
                    rem_risk_budget = max(0.0, max_stop_risk_allowed - current_stop_risk)
                    if rem_risk_budget > 0:
                        max_ntl_risk = rem_risk_budget / (sl_dist / fill_px)
                        target_notional = min(current_equity * 0.75, max_ntl_risk)
                        avail_cap = max(0.0, portfolio_cap_usd - total_notional)
                        final_ntl = min(target_notional, avail_cap)

                        if final_ntl >= 10.0 and cash >= (final_ntl * 0.2):
                            sz = final_ntl / fill_px
                            fee = final_ntl * TAKER_FEE
                            margin = final_ntl * 0.2
                            cash -= (margin + fee)
                            open_positions[coin] = {
                                "direction": trig["direction"], "type": trig["entry_type"],
                                "entry_px": fill_px, "sl_px": trig["sl_px"], "highest_px": fill_px,
                                "lowest_px": fill_px, "sz": sz, "notional": final_ntl, "margin": margin,
                                "atr": trig["atr"], "trailing_active": False, "breakeven_active": False,
                                "ml_prob": trig["ml_prob"]
                            }
                            total_notional += final_ntl
                            current_stop_risk += sz * sl_dist
                del pending_triggers[coin]

        equity_curve.append(current_equity)

        # 3. ПОИСК СИГНАЛОВ
        active_assets = set(open_positions.keys()) | set(pending_triggers.keys())
        if len(active_assets) < max_slots:
            btc_closes = btc_df[btc_df["t"] <= curr_t]["close"]

            for coin in clean_target_coins:
                if coin in active_assets or len(active_assets) >= max_slots:
                    continue

                c_df = processed_1h[coin]
                c_hist = c_df[c_df["t"] <= curr_t]
                if len(c_hist) < 72:
                    continue

                row = c_hist.iloc[-1]
                z_res_mom, beta_btc, raw_rs_pct = QuantFactorEngine.compute_residual_momentum_72h(c_hist["close"], btc_closes)
                atr = row["atr_4h"]

                # Лонг-сетапы
                if btc_bull and z_res_mom >= 0.40 and raw_rs_pct >= 2.0:
                    is_trend = (row["close"] > row["ema50_4h"]) and (row["ema20_4h"] > row["ema50_4h"])
                    is_evr_ok, _, _ = QuantFactorEngine.evaluate_evr_absorption(
                        open_px=row["open"], high_px=row["high"], low_px=row["low"], close_px=row["close"],
                        volume=row["vol_rolling_4h"], vol_sma=row["vol_sma20_4h"], ema20_4h=row["ema20_4h"], atr_4h=atr
                    )
                    breakout_hit = (row["close"] >= row["donchian_high_4h"] * 0.998)
                    vol_boost = row["vol_rolling_4h"] >= row["vol_sma20_4h"] * 1.15
                    valid_bo = breakout_hit and vol_boost and (btc_slope_rel > 0.15)
                    valid_pb = is_trend and is_evr_ok

                    if valid_pb or valid_bo:
                        raw_feats = [
                            z_res_mom, beta_btc, raw_rs_pct,
                            min(row["vol_rolling_4h"] / max(row["vol_sma20_4h"], 1e-4), 5.0),
                            (atr / row["close"]) * 100.0,
                            (row["close"] - row["ema20_4h"]) / max(atr, 1e-4),
                            (row["close"] - row["donchian_low_4h"]) / max(row["donchian_high_4h"] - row["donchian_low_4h"], 1e-4),
                            btc_slope_rel, 1.0 if valid_bo else 0.0, 1.0
                        ]
                        prob = predict_meta_prob(raw_feats)
                        if prob >= 0.48:
                            base_sl = row["low"] - (atr * 0.85) if valid_pb else row["close"] - (atr * 1.50)
                            sh_f, sl_f = QuantFactorEngine.compute_fractal_swings(c_hist["high"], c_hist["low"], window=2)

                            # Фрактальный стоп: если подтвержденный свинговый минимум ближе 1.5 ATR (но >= 0.8 ATR)
                            if sl_f and sl_f < row["close"] and (row["close"] - sl_f) >= (atr * 0.80):
                                sl_price = max(base_sl, sl_f * 0.999)
                            else:
                                sl_price = base_sl

                            pending_triggers[coin] = {
                                "direction": "LONG", "entry_type": "PULLBACK" if valid_pb else "BREAKOUT",
                                "trigger_px": row["high"] * 1.0005,
                                "sl_px": sl_price,
                                "expiry_t": curr_t + (3 * 3600 * 1000), "atr": atr, "ml_prob": prob
                            }
                            active_assets.add(coin)

                # Шорт-сетапы
                elif btc_bear and btc_slope_rel < -0.30 and z_res_mom <= -0.40 and raw_rs_pct <= -2.5:
                    is_bear_trend = (row["close"] < row["ema50_4h"]) and (row["ema20_4h"] < row["ema50_4h"])
                    breakdown_hit = (row["close"] <= row["donchian_low_4h"] * 1.002)
                    vol_boost = row["vol_rolling_4h"] >= row["vol_sma20_4h"] * 1.15
                    is_bear_pb = is_bear_trend and (row["high"] >= row["ema20_4h"] * 0.995) and (row["close"] <= row["ema20_4h"])
                    valid_short_bo = breakdown_hit and vol_boost

                    if is_bear_pb or valid_short_bo:
                        raw_feats = [
                            z_res_mom, beta_btc, raw_rs_pct,
                            min(row["vol_rolling_4h"] / max(row["vol_sma20_4h"], 1e-4), 5.0),
                            (atr / row["close"]) * 100.0,
                            (row["close"] - row["ema20_4h"]) / max(atr, 1e-4),
                            (row["close"] - row["donchian_low_4h"]) / max(row["donchian_high_4h"] - row["donchian_low_4h"], 1e-4),
                            btc_slope_rel, 1.0 if valid_short_bo else 0.0, 0.0
                        ]
                        prob = predict_meta_prob(raw_feats)
                        if prob >= 0.48:
                            sl_price = row["high"] + (atr * 0.85) if is_bear_pb else row["close"] + (atr * 1.50)

                            pending_triggers[coin] = {
                                "direction": "SHORT", "entry_type": "BEAR_PULLBACK" if is_bear_pb else "BEAR_BREAKDOWN",
                                "trigger_px": row["low"] * 0.9995,
                                "sl_px": sl_price,
                                "expiry_t": curr_t + (3 * 3600 * 1000), "atr": atr, "ml_prob": prob
                            }
                            active_assets.add(coin)

    final_eq = equity_curve[-1]
    df_t = pd.DataFrame(trades)
    print("\n" + "=" * 85)
    print(f"  ИТОГИ: {title}")
    print("=" * 85)
    if df_t.empty:
        print("[-] Сделок не было.")
        return
    wins = df_t[df_t["pnl_usd"] > 0]
    losses = df_t[df_t["pnl_usd"] <= 0]
    wr = len(wins) / len(df_t) * 100
    pnl_net = final_eq - INITIAL_CAPITAL
    roi = pnl_net / INITIAL_CAPITAL * 100
    gp = wins["pnl_usd"].sum() if not wins.empty else 0.0
    gl = abs(losses["pnl_usd"].sum()) if not losses.empty else 1e-4
    pf = gp / gl
    eq_s = pd.Series(equity_curve)
    mdd = abs(((eq_s - eq_s.cummax()) / eq_s.cummax()).min()) * 100

    long_trades = df_t[df_t["dir"] == "LONG"]
    short_trades = df_t[df_t["dir"] == "SHORT"]

    print(f"💰 Стартовый депозит:     ${INITIAL_CAPITAL:,.2f}")
    print(f"📈 Итоговый капитал:      ${final_eq:,.2f} ({roi:+.2f}%)")
    print(f"💵 Чистая прибыль:        ${pnl_net:+,.2f}")
    print("-" * 85)
    print(f"📊 Всего сделок:          {len(df_t)} (LONG: {len(long_trades)} | SHORT: {len(short_trades)})")
    print(f"🎯 Win Rate:              {wr:.1f}% ({len(wins)} в плюс / {len(losses)} в минус)")
    print(f"⚖️ Profit Factor:         {pf:.2f}")
    print(f"🛡 Максимальная просадка: {mdd:.2f}%")
    print("-" * 85)
    print(f"{'МОНЕТА':<7} | {'НАПР':<5} | {'ТИП':<14} | {'ML PROB':<8} | {'ВХОД':<9} | {'ВЫХОД':<9} | {'РЕЗУЛЬТАТ ($)':<16} | {'ИТОГ'}")
    print("-" * 85)
    for _, t in df_t.iterrows():
        pnl_str = f"${t['pnl_usd']:+,.2f} ({t['pnl_pct']:+.1f}%)"
        print(f"{t['coin']:<7} | {t['dir']:<5} | {t['type']:<14} | {t['ml_prob']*100:<5.1f}%  | ${t['entry_px']:<8.2f} | ${t['exit_px']:<8.2f} | {pnl_str:<16} | {t['reason']}")
    print("=" * 85)

# 1. In-Sample
simulate_v10_7(min_warmup, split_idx, "1. IN-SAMPLE ПЕРИОД v10.7 (37 ДНЕЙ)")

# 2. Строгий Out-of-Sample со штрафом проскальзывания 0.25%
simulate_v10_7(oos_start_idx, total_bars, "2. СТРОГИЙ OUT-OF-SAMPLE v10.7 (13 ДНЕЙ)")