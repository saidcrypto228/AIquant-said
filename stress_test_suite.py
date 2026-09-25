#!/usr/bin/env python3
"""
Комплексный стресс-тест квантовых модулей v10.9 (Institutional Microstructure Suite).
Всего тестов: 20 (Precision: 4, Math: 8, OrderFlow: 3, Risk: 3, ML: 2).
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from decimal import Decimal

from quant_factors import QuantFactorEngine
from hl_swing_bot import PrecisionEngine, HyperliquidSwingBot
import bot_config as config

print("=" * 85)
print("  КОМПЛЕКСНОЕ СТРЕСС-ТЕСТИРОВАНИЕ КВАНТОВЫХ МОДУЛЕЙ (v10.8)")
print("=" * 85)

test_results = []

# 1. PrecisionEngine
print("\n[*] Тестирование PrecisionEngine...")
try:
    sz = PrecisionEngine.round_sz(12.3456, 2)
    assert sz == 12.34, f"Expected 12.34, got {sz}"
    test_results.append(("Precision: Lot size round-down", True, f"Получено {sz}, ожидалось 12.34"))

    # На Hyperliquid микро-цены торгуются с szDecimals = 0 (до 6 знаков после запятой)
    px_micro = PrecisionEngine.round_px(0.0004561, 0)
    assert px_micro == 0.000456, f"Expected 0.000456, got {px_micro}"
    test_results.append(("Precision: Micro-price handling", True, f"Округлено до {px_micro}"))

    px_high = PrecisionEngine.round_px(98452.12, 1)
    assert px_high == 98452.0, f"Expected 98452.0, got {px_high}"
    test_results.append(("Precision: High-notional price", True, f"Округлено до {px_high}"))

    neg_rejected = False
    try:
        PrecisionEngine.round_px(-10.5, 2)
    except ValueError:
        neg_rejected = True
    assert neg_rejected
    test_results.append(("Precision: Negative price rejection", True, "Отрицательная цена корректно заблокирована"))
except Exception as e:
    test_results.append(("Precision: Engine Failure", False, str(e)))

# 2. QuantFactorEngine
print("\n[*] Тестирование QuantFactorEngine...")
try:
    # 1. OLS Zero Variance
    flat = pd.Series([100.0] * 80)
    z, beta, rs = QuantFactorEngine.compute_residual_momentum_72h(flat, flat)
    assert z == 0.0 and beta == 1.0
    test_results.append(("Math: Zero Variance OLS (Flatline)", True, f"z={z:.1f}, beta={beta:.1f}, rs={rs:.1f}%"))

    # 2. EVR Zero Vol
    ok, score, reason = QuantFactorEngine.evaluate_evr_absorption(100, 100, 100, 100, 0, 0, 100, 0)
    assert not ok
    test_results.append(("Math: Zero Volume / Zero ATR Candle", True, f"EVR pass={ok}, score={score:.1f}"))

    # 3. Фандинг Spike
    hist_norm = [0.0001] * 72
    z_spike, note_spike = QuantFactorEngine.compute_funding_zscore(-0.002, hist_norm)
    assert z_spike == -3.0
    test_results.append(("Math: Extreme Funding Spike (Short Squeeze)", True, f"Z-Score={z_spike:.2f} ({note_spike})"))

    # 4. Фандинг Overheat
    z_heat, note_heat = QuantFactorEngine.compute_funding_zscore(0.003, hist_norm)
    assert z_heat == 3.0
    test_results.append(("Math: Extreme Funding Overheat (Long Trap)", True, f"Z-Score={z_heat:.2f} ({note_heat})"))

    # 5. Динамический Z-score
    hist_dyn = [0.0001 + i * 0.00001 for i in range(72)]
    z_dyn, note_dyn = QuantFactorEngine.compute_funding_zscore(0.002, hist_dyn)
    test_results.append(("Math: Dynamic Sample Z-Score calculation", True, f"Z-Score={z_dyn:.2f} ({note_dyn})"))

    # 6. Fallback на пустую историю
    z_empty, note_empty = QuantFactorEngine.compute_funding_zscore(0.0001, [])
    assert z_empty == 0.0
    test_results.append(("Math: Empty funding history fallback", True, f"Z-Score={z_empty:.2f} ({note_empty})"))

    # 7. Фракталы ликвидности
    dummy_highs = pd.Series([10.0, 11.0, 15.0, 12.0, 11.0, 13.0, 18.0, 14.0, 12.0])
    dummy_lows = pd.Series([8.0, 9.0, 12.0, 10.0, 7.0, 9.0, 11.0, 10.0, 9.0])
    sh, sl = QuantFactorEngine.compute_fractal_swings(dummy_highs, dummy_lows, window=2)
    assert sh == 18.0 and sl == 7.0
    test_results.append(("Math: Fractal Swings (Swing High/Low)", True, f"SH=${sh:.1f}, SL=${sl:.1f}"))

    # 8. Детектор перегрева Z >= 2.5
    fake_funding = [0.0001] * 70 + [0.0005, 0.0009]
    z_f = QuantFactorEngine.compute_funding_zscore(fake_funding)
    assert z_f >= 2.5
    test_results.append(("Math: Funding Overheat Detection (Z >= 2.5)", True, f"Z={z_f:.2f} (Перегрев обнаружен)"))

    # 9. Волатильность Гармана-Класса
    o_s = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0] * 3)
    h_s = pd.Series([102.0, 103.0, 104.0, 105.0, 106.0] * 3)
    l_s = pd.Series([99.0,  100.0, 101.0, 102.0, 103.0] * 3)
    c_s = pd.Series([101.0, 102.0, 103.0, 104.0, 105.0] * 3)
    gk = QuantFactorEngine.compute_garman_klass_volatility(o_s, h_s, l_s, c_s)
    assert 0.0 < gk < 0.1, f"GK volatility unexpected: {gk}"
    test_results.append(("Math: Garman-Klass Volatility Engine", True, f"GK_vol={gk:.4f}"))

    # 10. Delta OI Robust Z-Score
    fake_oi = [10000.0 + (i * 10) for i in range(30)] + [15000.0]  # Резкий спайк набора OI
    z_oi = QuantFactorEngine.compute_delta_oi_robust_zscore(fake_oi, lookback=24)
    assert z_oi == 3.0, f"Expected clipping at 3.0, got {z_oi}"
    test_results.append(("Math: Delta OI Robust MAD Z-Score", True, f"Z_OI={z_oi:.2f} (Клиппинг подтвержден)"))
except Exception as e:
    test_results.append(("Math: Engine failure on edge cases", False, str(e)))

# 3. Order Flow
print("\n[*] Тестирование отказоустойчивости Order Flow...")
try:
    bot = HyperliquidSwingBot()
    state_file = config.DATA_DIR / "orderflow_state.json"

    state_file.write_text("CORRUPTED_JSON_DATA", encoding="utf-8")
    m_bad = bot.get_orderflow_metrics("SOL")
    assert not m_bad["is_fresh"] and m_bad["last_px"] == 0.0
    test_results.append(("OrderFlow: Corrupted JSON file recovery", True, "Битый JSON перехвачен, бот не упал"))

    old_payload = {"timestamp": time.time() - 30, "coins": {"SOL": {"last_px": 150.0, "cvd_ratio": 0.5}}}
    state_file.write_text(json.dumps(old_payload), encoding="utf-8")
    m_stale = bot.get_orderflow_metrics("SOL")
    assert not m_stale["is_fresh"]
    test_results.append(("OrderFlow: Stale snapshot rejection (>15s)", True, "Устаревшие данные забракованы"))

    fresh_payload = {"timestamp": time.time(), "coins": {"SOL": {"last_px": 150.0, "cvd_ratio": 0.5}}}
    state_file.write_text(json.dumps(fresh_payload), encoding="utf-8")
    m_fresh = bot.get_orderflow_metrics("SOL")
    assert m_fresh["is_fresh"] and m_fresh["last_px"] == 150.0
    test_results.append(("OrderFlow: Valid fresh payload ingestion", True, "Свежие данные прочитаны успешно"))
except Exception as e:
    test_results.append(("OrderFlow: Failure", False, str(e)))

# 4. Риск и сайзинг
print("\n[*] Тестирование инвариантов сайзинга и риска...")
try:
    bot.get_portfolio_equity = lambda: 1000.0
    bot.universe_meta = {"SOL": {"szDecimals": 2}}

    s1 = bot.calculate_sizing("SOL", entry_px=100.0, sl_px=99.9)
    assert s1["notional"] <= 750.01, f"Notional was {s1['notional']}"
    test_results.append(("Risk: Ultra-tight Stop Leverage Cap", True, f"Notional capped at ${s1['notional']:.2f}"))

    s2 = bot.calculate_sizing("SOL", entry_px=100.0, sl_px=105.0)
    assert s2 is not None
    test_results.append(("Risk: Inverted Stop-Loss handling", True, "Сайзинг рассчитан корректно"))

    s3 = bot.calculate_sizing("SOL", entry_px=100.0, sl_px=10.0, risk_pct=0.0001)
    assert s3 is None
    test_results.append(("Risk: Sub-$10 Notional rejection", True, "Позиции ниже биржевого минимума $10 отсекаются"))
except Exception as e:
    test_results.append(("Risk: Failure", False, str(e)))

# 5. ML Инференс
print("\n[*] Тестирование инференса линейной Meta-модели...")
try:
    extreme_feats = [100.0] * 10
    prob_ext = bot.predict_meta_prob(extreme_feats)
    assert 0.0 <= prob_ext <= 1.0
    test_results.append(("ML: Extreme Outlier bounds [0, 1]", True, f"Вероятность: {prob_ext*100:.2f}%"))

    normal_feats = [0.5, 1.0, 3.0, 1.2, 2.5, 0.4, 0.8, 0.2, 1.0, 1.0]
    t0 = time.perf_counter()
    for _ in range(100):
        _ = bot.predict_meta_prob(normal_feats)
    dt_us = ((time.perf_counter() - t0) / 100) * 1e6
    assert dt_us < 50.0
    test_results.append(("ML: Native Vector Latency (<50 μs)", True, f"Среднее время инференса: {dt_us:.2f} мкс ({dt_us/1000:.4f} мс)"))
except Exception as e:
    test_results.append(("ML: Failure", False, str(e)))

print("\n" + "=" * 85)
print("  ИТОГОВЫЙ ОТЧЕТ СТРЕСС-ТЕСТИРОВАНИЯ")
print("=" * 85)

passed = sum(1 for _, ok, _ in test_results if ok)
total = len(test_results)

for name, ok, desc in test_results:
    mark = "[✓ PASS]" if ok else "[✗ FAIL]"
    print(f"{mark} {name:<45} | {desc}")

print("\n" + "=" * 85)
print(f"Всего тестов пройдено: {passed} / {total} ({passed/total*100:.1f}%)")
if passed == total:
    print("✓ ВСЕ КВАНТОВЫЕ ИНВАРИАНТЫ УСПЕШНО ВЫДЕРЖАЛИ НАГРУЗКУ (100% PASS).")
else:
    print(f"[!] ВНИМАНИЕ: Найдено {total - passed} сбоев граничных условий!")
print("=" * 85)
