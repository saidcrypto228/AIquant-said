#!/usr/bin/env python3
"""
Автоматизированный стресс-тест квантовой архитектуры v9.3.
- Тестирование нативного инференса линейной модели v9.2 (predict_meta_prob).
- Проверка инвариантов сайзинга, точности котировок, устойчивости Order Flow и фандинга.
"""

import sys
import time
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

import bot_config as config
from quant_factors import QuantFactorEngine
from hl_swing_bot import PrecisionEngine, HyperliquidSwingBot

print("=" * 85)
print("  КОМПЛЕКСНОЕ СТРЕСС-ТЕСТИРОВАНИЕ КВАНТОВЫХ МОДУЛЕЙ (v9.3)")
print("=" * 85)

results = []

def record(test_name: str, passed: bool, details: str):
    status = "✓ PASS" if passed else "✗ FAIL"
    results.append({"name": test_name, "status": status, "details": details})
    print(f"[{status}] {test_name:<45} | {details}")

# 1. ТЕСТЫ PRECISION ENGINE
print("\n[*] Тестирование PrecisionEngine...")
try:
    sz_test = PrecisionEngine.round_sz(12.34567, sz_decimals=2)
    record("Precision: Lot size round-down", sz_test == 12.34, f"Получено {sz_test}, ожидалось 12.34")

    px_micro = PrecisionEngine.round_px(0.000456789, sz_decimals=0)
    record("Precision: Micro-price handling", px_micro > 0 and not math.isnan(px_micro), f"Округлено до {px_micro}")

    px_macro = PrecisionEngine.round_px(98452.123, sz_decimals=3)
    record("Precision: High-notional price", px_macro == 98452.0, f"Округлено до {px_macro}")

    try:
        PrecisionEngine.round_px(-10.5, sz_decimals=2)
        record("Precision: Negative price rejection", False, "Исключение не было выброшено!")
    except ValueError:
        record("Precision: Negative price rejection", True, "Отрицательная цена корректно заблокирована")
except Exception as e:
    record("Precision: Unexpected exception", False, str(e))

# 2. ТЕСТЫ QUANT FACTOR ENGINE
print("\n[*] Тестирование QuantFactorEngine...")
try:
    flat_series_1 = pd.Series([10.0] * 72)
    flat_series_2 = pd.Series([50000.0] * 72)
    z_mom, beta, rs = QuantFactorEngine.compute_residual_momentum_72h(flat_series_1, flat_series_2)
    record("Math: Zero Variance OLS (Flatline)", not math.isnan(z_mom) and not math.isnan(beta), f"z={z_mom}, beta={beta}, rs={rs}%")

    is_evr, score, m = QuantFactorEngine.evaluate_evr_absorption(
        open_px=100.0, high_px=100.0, low_px=100.0, close_px=100.0,
        volume=0.0, vol_sma=0.0, ema20_4h=100.0, atr_4h=0.0
    )
    record("Math: Zero Volume / Zero ATR Candle", not is_evr and not math.isnan(score), f"EVR pass={is_evr}, score={score}")

    z_fund_neg, is_ok_neg = QuantFactorEngine.compute_funding_zscore(-120.0)
    record("Math: Extreme Funding Spike (Short Squeeze)", z_fund_neg <= -3.0, f"Z-Score={z_fund_neg:.2f} (Защитный клиппинг)")

    z_fund_pos, is_ok_pos = QuantFactorEngine.compute_funding_zscore(+120.0)
    record("Math: Extreme Funding Overheat (Long Trap)", z_fund_pos >= 3.0, f"Z-Score={z_fund_pos:.2f} (Защитный клиппинг)")

    z_dyn, _ = QuantFactorEngine.compute_funding_zscore(30.0, funding_history=[10.0, 12.0, 11.0, 10.5])
    record("Math: Dynamic Sample Z-Score calculation", z_dyn > 2.0, f"Z-Score={z_dyn:.2f} (Успешно рассчитан)")

    z_empty, _ = QuantFactorEngine.compute_funding_zscore(11.0, funding_history=[])
    record("Math: Empty funding history fallback", abs(z_empty) < 0.01, f"Z-Score={z_empty:.2f} (Нейтральный фандинг)")
except Exception as e:
    record("Math: Engine failure on edge cases", False, str(e))

# 3. ТЕСТЫ ОТКАЗОУСТОЙЧИВОСТИ ORDER FLOW
print("\n[*] Тестирование отказоустойчивости Order Flow...")
state_path = config.DATA_DIR / "orderflow_state.json"
backup_state = None
if state_path.exists():
    backup_state = state_path.read_text(encoding="utf-8")

bot_instance = HyperliquidSwingBot()

try:
    state_path.write_text('{"timestamp": 12345, "coins": { "SOL": { "cvd_ratio": ', encoding="utf-8")
    metrics_corrupt = bot_instance.get_orderflow_metrics("SOL")
    record("OrderFlow: Corrupted JSON file recovery", metrics_corrupt["is_fresh"] is False, "Битый JSON перехвачен, бот не упал")

    stale_payload = {
        "timestamp": time.time() - 60.0,
        "coins": {"SOL": {"cvd_ratio": 0.85, "cvd_usd": 50000.0}}
    }
    state_path.write_text(json.dumps(stale_payload), encoding="utf-8")
    metrics_stale = bot_instance.get_orderflow_metrics("SOL")
    record("OrderFlow: Stale snapshot rejection (>15s)", metrics_stale["is_fresh"] is False, "Устаревшие данные забракованы")

    fresh_payload = {
        "timestamp": time.time(),
        "coins": {"SOL": {"cvd_ratio": 0.45, "cvd_usd": 12000.0}}
    }
    state_path.write_text(json.dumps(fresh_payload), encoding="utf-8")
    metrics_fresh = bot_instance.get_orderflow_metrics("SOL")
    record("OrderFlow: Valid fresh payload ingestion", metrics_fresh["is_fresh"] is True and metrics_fresh["cvd_ratio"] == 0.45, "Свежие данные прочитаны успешно")
finally:
    if backup_state is not None:
        state_path.write_text(backup_state, encoding="utf-8")

# 4. ТЕСТЫ РИСК-МЕНЕДЖМЕНТА И САЙЗИНГА
print("\n[*] Тестирование инвариантов сайзинга и риска...")
try:
    sz_spike = bot_instance.calculate_sizing("SOL", entry_px=100.0, sl_px=99.999, risk_pct=0.015)
    record("Risk: Ultra-tight Stop Leverage Cap", sz_spike["notional"] <= 1200.01, f"Notional capped at ${sz_spike['notional']:.2f}")

    sz_inverted = bot_instance.calculate_sizing("SOL", entry_px=100.0, sl_px=105.0, risk_pct=0.015)
    record("Risk: Inverted Stop-Loss handling", sz_inverted is not None, "Сайзинг рассчитан корректно")

    sz_dust = bot_instance.calculate_sizing("SOL", entry_px=100.0, sl_px=90.0, risk_pct=0.0001)
    record("Risk: Sub-$10 Notional rejection", sz_dust is None, "Позиции ниже биржевого минимума $10 отсекаются")
except Exception as e:
    record("Risk: Sizing Engine failure", False, str(e))

# 5. ТЕСТЫ REGULARIZED LINEAR META-MODEL INFERENCE (v9.2)
print("\n[*] Тестирование инференса линейной Meta-модели...")
if getattr(bot_instance, "meta_weights", False):
    try:
        # Вектор с экстремальными выбросами (10 квантовых признаков)
        outlier_raw = [50.0, 15.0, 250.0, 100.0, 50.0, 20.0, 1.5, 10.0, 1.0, 1.0]

        prob = bot_instance.predict_meta_prob(outlier_raw)
        record("ML: Extreme Outlier bounds [0, 1]", 0.0 <= prob <= 1.0, f"Вероятность: {prob*100:.2f}%")

        # Замер задержки нативного инференса (скалярное произведение + сигмоида)
        t_start = time.perf_counter()
        for _ in range(5000):
            _ = bot_instance.predict_meta_prob(outlier_raw)
        avg_latency_us = ((time.perf_counter() - t_start) / 5000) * 1_000_000
        record("ML: Native Vector Latency (<50 μs)", avg_latency_us < 50.0, f"Среднее время инференса: {avg_latency_us:.2f} мкс (0.{int(avg_latency_us*1000):03d} мс)")
    except Exception as e:
        record("ML: Inference crashed on edge case", False, str(e))
else:
    record("ML: Model availability", False, "Модель data/meta_model.json не загружена")

# ИТОГОВЫЙ СРЕЗ
print("\n" + "=" * 85)
print("  ИТОГОВЫЙ ОТЧЕТ СТРЕСС-ТЕСТИРОВАНИЯ")
print("=" * 85)
total = len(results)
passed_cnt = sum(1 for r in results if "PASS" in r["status"])
failed_cnt = total - passed_cnt

print(f"Всего тестов пройдено: {passed_cnt} / {total} ({passed_cnt/total*100:.1f}%)")
if failed_cnt > 0:
    print(f"[!] ВНИМАНИЕ: Найдено {failed_cnt} сбоев граничных условий!")
else:
    print("✓ ВСЕ КВАНТОВЫЕ ИНВАРИАНТЫ УСПЕШНО ВЫДЕРЖАЛИ НАГРУЗКУ (100% PASS).")
print("=" * 85)
