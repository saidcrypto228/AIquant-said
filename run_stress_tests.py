#!/usr/bin/env python3
"""
Institutional Stress-Testing Suite for 4H Swing Trend Engine.
1. Monte Carlo Simulation (1,000 iterations): Drawdown VaR 95/99% and Risk of Ruin.
2. Parameter Sensitivity Grid (SL ATR 1.4 - 2.2, EMA 16 - 24): Detecting Overfitting Peaks.
3. Friction & Adverse Slippage Stress: Taker Fees + 0.20% Slippage.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/history_deep")
TRADES_FILE = Path("results/refined_swing_trades.csv")

def load_resampled_4h(coin: str) -> pd.DataFrame:
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
    df_4h["coin"] = coin
    df_4h["log_ret_72h"] = np.log(c / c.shift(18))
    return df_4h.dropna().reset_index(drop=True)

# -------------------------------------------------------------
# СТРЕСС-ТЕСТ 1: МОНТЕ-КАРЛО (1,000 СИМУЛЯЦИЙ ПЕРЕСТАНОВКИ СДЕЛОК)
# -------------------------------------------------------------
def run_monte_carlo(trades_pnl: list, initial_cap: float = 200.0, iters: int = 1000):
    print("\n" + "=" * 65)
    print("  ТЕСТ 1: МОНТЕ-КАРЛО СИМУЛЯЦИЯ (1,000 СЦЕНАРИЕВ СУДЬБЫ)")
    print("=" * 65)

    n_trades = len(trades_pnl)
    if n_trades < 20:
        print("[-] Недостаточно сделок для Монте-Карло.")
        return

    pnl_arr = np.array(trades_pnl)
    max_dds = []
    end_equities = []
    ruin_count = 0

    for _ in range(iters):
        # Случайное бутстрап-перемешивание с возвращением
        shuffled = np.random.choice(pnl_arr, size=n_trades, replace=True)
        curve = [initial_cap]
        cur_eq = initial_cap
        ruined = False

        for p in shuffled:
            cur_eq += p
            if cur_eq <= initial_cap * 0.50: # Просадка 50% = банкротство для теста
                ruined = True
            curve.append(cur_eq)

        if ruined:
            ruin_count += 1

        curve = np.array(curve)
        peaks = np.maximum.accumulate(curve)
        dds = (peaks - curve) / peaks
        max_dds.append(dds.max() * 100.0)
        end_equities.append(cur_eq)

    max_dds = np.array(max_dds)
    end_equities = np.array(end_equities)

    median_dd = np.median(max_dds)
    dd_95 = np.percentile(max_dds, 95)
    dd_99 = np.percentile(max_dds, 99)
    median_eq = np.median(end_equities)
    eq_5th = np.percentile(end_equities, 5)

    print(f"Медианная просадка (типичный случай) : -{median_dd:.2f}%")
    print(f"Худшая просадка с надежностью 95%    : -{dd_95:.2f}% (VaR 95%)")
    print(f"Худшая просадка с надежностью 99%    : -{dd_99:.2f}% (VaR 99%)")
    print(f"Медианный конечный капитал           : ${median_eq:.2f}")
    print(f"Конечный капитал в худших 5% случаев: ${eq_5th:.2f}")
    print(f"Вероятность потери >50% депозита     : {ruin_count / iters * 100:.2f}%")

    if dd_95 <= 22.0 and ruin_count == 0:
        print("[✓] ВЕРДИКТ МОНТЕ-КАРЛО: PASS (Система устойчива к перестановкам)")
    else:
        print("[!] ВЕРДИКТ МОНТЕ-КАРЛО: WARN / FAIL (Худшие сценарии опасны)")

# -------------------------------------------------------------
# СТРЕСС-ТЕСТ 2: ТЕСТ НА ЧУВСТВИТЕЛЬНОСТЬ ПАРАМЕТРОВ (ПЛАТО)
# -------------------------------------------------------------
def run_parameter_sensitivity(btc_df: pd.DataFrame, sol_df: pd.DataFrame, near_df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  ТЕСТ 2: АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ (ПАРАМЕТРИЧЕСКОЕ ПЛАТО)")
    print("=" * 65)

    sl_mults = [1.40, 1.60, 1.80, 2.00, 2.20]
    ema_spans = [16, 18, 20, 22, 24]

    results = []

    for ema_p in ema_spans:
        # Пересчитываем EMA для всех
        b_df = btc_df.copy()
        b_df["ema_fast"] = b_df["close"].ewm(span=ema_p, adjust=False).mean()
        b_df["ema_slow"] = b_df["close"].ewm(span=50, adjust=False).mean()
        b_df["btc_bull"] = (b_df["close"] > b_df["ema_slow"]) & (b_df["ema_fast"] > b_df["ema_slow"])
        btc_dict = b_df.set_index("datetime")[["btc_bull", "log_ret_72h"]].to_dict(orient="index")

        for sl_m in sl_mults:
            # Симулируем SOL и NEAR
            total_pnl = 0.0
            trade_count = 0
            wins = 0

            for c_raw in [sol_df, near_df]:
                df = c_raw.copy()
                df["ema_fast"] = df["close"].ewm(span=ema_p, adjust=False).mean()
                df["ema_slow"] = df["close"].ewm(span=50, adjust=False).mean()
                df["btc_bull"] = df["datetime"].map(lambda dt: btc_dict.get(dt, {}).get("btc_bull", False))
                df["btc_ret"] = df["datetime"].map(lambda dt: btc_dict.get(dt, {}).get("log_ret_72h", 0.0))
                df["rs"] = df["log_ret_72h"] - df["btc_ret"]

                in_pos = False
                entry_p = 0.0
                sl_p = 0.0
                dur = 0
                pos_usd = 30.0 # Фиксированный объем $30 для чистоты сравнения параметров

                for idx in range(len(df)):
                    row = df.iloc[idx]
                    if not in_pos:
                        if row["btc_bull"] and row["close"] > row["ema_slow"] and row["ema_fast"] > row["ema_slow"] and row["rs"] > 0:
                            limit_p = row["ema_fast"]
                            if row["low"] <= limit_p <= row["high"]:
                                in_pos = True
                                entry_p = limit_p
                                sl_p = limit_p - (row["atr_14"] * sl_m)
                                dur = 0
                    else:
                        dur += 1
                        exit_p = None
                        if row["low"] <= sl_p:
                            exit_p = sl_p
                        elif dur >= 2 and row["close"] < row["ema_fast"]:
                            exit_p = row["close"]

                        if exit_p is not None:
                            ret = (exit_p / entry_p) - 1.0 - 0.0007
                            pnl = pos_usd * ret
                            total_pnl += pnl
                            trade_count += 1
                            if pnl > 0:
                                wins += 1
                            in_pos = False

            results.append({
                "EMA": ema_p, "SL_ATR": sl_m,
                "Trades": trade_count, "PnL_$": round(total_pnl, 1),
                "WinRate": round(wins / max(trade_count, 1) * 100, 1)
            })

    res_df = pd.DataFrame(results)
    pivot_pnl = res_df.pivot(index="EMA", columns="SL_ATR", values="PnL_$")
    print("\nТаблица PnL ($) при фиксированном объеме $30 (сетка 5x5):")
    print(pivot_pnl.to_string())

    min_val = pivot_pnl.min().min()
    negative_cells = (pivot_pnl < 0).sum().sum()
    print(f"\nОтрицательных ячеек в сетке: {negative_cells} из 25")
    print(f"Худший результат в сетке    : ${min_val:.1f}")

    if negative_cells <= 3 and min_val > -5.0:
        print("[✓] ВЕРДИКТ РОБАСТНОСТИ: PASS (Наблюдается широкое плато прибыльности)")
    else:
        print("[!] ВЕРДИКТ РОБАСТНОСТИ: FAIL (Стратегия чувствительна к параметрам)")

# -------------------------------------------------------------
# СТРЕСС-ТЕСТ 3: УДВОЕНИЕ КОМИССИЙ И ПРОСКАЛЬЗЫВАНИЕ
# -------------------------------------------------------------
def run_friction_stress(trades_csv: Path, initial_cap: float = 200.0):
    print("\n" + "=" * 65)
    print("  ТЕСТ 3: СТРЕСС-ТЕСТ КОМИССИЙ И СКОЛЬЖЕНИЯ (WORST FRICTION)")
    print("=" * 65)

    if not trades_csv.exists():
        print(f"[-] Файл {trades_csv} не найден.")
        return

    df_t = pd.read_csv(trades_csv)

    # Моделируем худшие рыночные условия:
    # 1. Вход маркетом вместо лимита: комиссия тейкера 0.05% вместо 0.02%
    # 2. Выход: дополнительное проскальзывание 0.20% на каждом стопе и закрытии
    # Суммарное дополнительное трение: ~0.23% от объема на сделку

    pos_usd_est = 35.0 # Средний размер позиции
    extra_penalty = pos_usd_est * 0.0023

    orig_pnl = df_t["pnl"].sum()
    stressed_pnls = df_t["pnl"] - extra_penalty
    stressed_sum = stressed_pnls.sum()

    stressed_curve = [initial_cap]
    c_eq = initial_cap
    for p in stressed_pnls:
        c_eq += p
        stressed_curve.append(c_eq)
    stressed_curve = np.array(stressed_curve)
    peaks = np.maximum.accumulate(stressed_curve)
    stressed_dd = ((peaks - stressed_curve) / peaks).max() * 100.0

    print(f"Оригинальный PnL стратегии       : +${orig_pnl:.2f}")
    print(f"PnL при двойных комиссиях и слипе: +\({stressed_sum:.2f} (потеря\){orig_pnl - stressed_sum:.2f} на трении)")
    print(f"Макс. просадка в жестких условиях: -{stressed_dd:.2f}% (было -14.34%)")

    if stressed_sum > 0 and stressed_dd < 22.0:
        print("[✓] ВЕРДИКТ ТРЕНИЯ: PASS (Запас прочности перекрывает комиссии и слипы)")
    else:
        print("[!] ВЕРДИКТ ТРЕНИЯ: FAIL (Комиссии сжирают все преимущество)")

def main():
    # 1. Запуск Монте-Карло
    # Читаем сделки из финального бэктеста
    # Заново прогоняем для чистоты лога, если файла нет
    trades_path = Path("results/final_4h_trades.csv")

    btc_df = load_resampled_4h("BTC")
    sol_df = load_resampled_4h("SOL")
    near_df = load_resampled_4h("NEAR")

    if not trades_path.exists():
        # Быстрый сбор сделок
        import run_4h_final
        # Запустит и создаст лог

    if trades_path.exists():
        df_tr = pd.read_csv(trades_path)
        run_monte_carlo(df_tr["pnl"].tolist())
        run_friction_stress(trades_path)
    else:
        print("[!] Файл final_4h_trades.csv пока не найден, пропускаем Монте-Карло.")

    run_parameter_sensitivity(btc_df, sol_df, near_df)
    print("\n" + "=" * 65)

if __name__ == "__main__":
    main()
