#!/usr/bin/env python3
"""
Institutional Monte-Carlo Stress Engine for Prop Challenge.
Simulates 10,000 randomized sequences of verified trades to measure ruin probability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRADES_PATH = Path("results/prop_trades_log.csv")
if not TRADES_PATH.exists():
    raise FileNotFoundError(f"Файл сделок {TRADES_PATH} не найден! Сначала запустите run_prop_strategy.py")

df_trades = pd.read_csv(TRADES_PATH)
trade_pnls = df_trades['pnl'].to_numpy(dtype=np.float64)
n_trades = len(trade_pnls)

print(f"[*] Загружено {n_trades} проверенных сделок для стресс-теста.")

N_SIMULATIONS = 10000
INITIAL_CAPITAL = 10000.0
MAX_ALLOWED_DD_PCT = 8.0

final_pnls = np.zeros(N_SIMULATIONS)
max_drawdowns = np.zeros(N_SIMULATIONS)
breached_count = 0

np.random.seed(42)

for sim in range(N_SIMULATIONS):
    # Случайная выборка сделок с возвращением (Bootstrap Resampling)
    sampled_pnls = np.random.choice(trade_pnls, size=n_trades, replace=True)

    # Расчет кривой капитала
    equity_curve = INITIAL_CAPITAL + np.cumsum(sampled_pnls)
    equity_curve = np.insert(equity_curve, 0, INITIAL_CAPITAL)

    # Расчет просадки от пика (Mark-to-Market Peak)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max * 100.0
    sim_max_dd = drawdowns.max()

    max_drawdowns[sim] = sim_max_dd
    final_pnls[sim] = (equity_curve[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0

    if sim_max_dd >= MAX_ALLOWED_DD_PCT:
        breached_count += 1

# Расчет квантилей риска
ruin_prob = (breached_count / N_SIMULATIONS) * 100.0
median_dd = np.median(max_drawdowns)
p95_dd = np.percentile(max_drawdowns, 95)
p99_dd = np.percentile(max_drawdowns, 99)
profitable_runs_pct = (final_pnls > 0).mean() * 100.0

print("=" * 65)
print(f"  РЕЗУЛЬТАТЫ СТРЕСС-ТЕСТА МОНТЕ-КАРЛО ({N_SIMULATIONS:,} СИМУЛЯЦИЙ)")
print("=" * 65)
print(f"Вероятность пробоя лимита 8.0% (Risk of Ruin): {ruin_prob:.2f}%")
print(f"Медианная просадка (50% квартиль)            : -{median_dd:.2f}%")
print(f"95% худшая просадка (VaR 95)                 : -{p95_dd:.2f}%")
print(f"99% наихудший сценарий (VaR 99)              : -{p99_dd:.2f}%")
print(f"Вероятность закончить период в плюс          : {profitable_runs_pct:.2f}%")
print(f"Средний ожидаемый PnL                        : {final_pnls.mean():+.2f}%")
print("=" * 65)

# Визуализация распределения
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(max_drawdowns, bins=50, color="#d62728", alpha=0.75, edgecolor="black")
plt.axvline(MAX_ALLOWED_DD_PCT, color="black", linestyle="--", linewidth=2, label="Prop Limit (8.0%)")
plt.axvline(p95_dd, color="orange", linestyle=":", linewidth=2, label=f"95% Worst ({p95_dd:.1f}%)")
plt.title("Распределение Max Drawdown (%)")
plt.xlabel("Max Drawdown (%)")
plt.ylabel("Количество симуляций")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(final_pnls, bins=50, color="#1f77b4", alpha=0.75, edgecolor="black")
plt.axvline(0, color="gray", linestyle="--", linewidth=1.5)
plt.axvline(final_pnls.mean(), color="green", linestyle="-", linewidth=2, label=f"Mean PnL ({final_pnls.mean():+.1f}%)")
plt.title("Распределение итогового PnL (%)")
plt.xlabel("PnL (%)")
plt.ylabel("Количество симуляций")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = Path("results/monte_carlo_distribution.png")
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"[+] График распределения сохранен в: {chart_path}")
