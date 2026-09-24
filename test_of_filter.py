#!/usr/bin/env python3
import json
from pathlib import Path
from quant_factors import QuantFactorEngine

state_file = Path("data/orderflow_state.json")
if not state_file.exists():
    print("[-] Файл orderflow_state.json не найден. Сначала запустите сборщик.")
    exit(1)

with open(state_file, "r", encoding="utf-8") as f:
    state = json.load(f)

print("=" * 85)
print("  ВАЛИДАЦИЯ ORDER FLOW ФИЛЬТРА: ПРОВЕРКА ИМПУЛЬСОВ НА ЖИВЫХ ДАННЫХ")
print("=" * 85)
print(f"{'АКТИВ':<7} | {'CVD ($)':<12} | {'CVD RATIO':<11} | {'ГИПОТЕЗА':<10} | {'ВЕРДИКТ ФИЛЬТРА МИКРОСТРУКТУРЫ'}")
print("-" * 85)

for coin, m in state.get("coins", {}).items():
    ratio = m["cvd_ratio"]
    cvd_usd = m["cvd_usd"]

    # Тест гипотезы Long
    is_long_ok, reason_long = QuantFactorEngine.verify_orderflow_confirmation("LONG", ratio, cvd_usd)
    # Тест гипотезы Short
    is_short_ok, reason_short = QuantFactorEngine.verify_orderflow_confirmation("SHORT", ratio, cvd_usd)

    cvd_str = f"${cvd_usd:+,.1f}"
    ratio_str = f"{ratio*100:+6.1f}%"

    if is_long_ok:
        verdict = f"🟢 ЛОНГ РАЗРЕШЕН  -> {reason_long}"
        hypo = "LONG"
    elif is_short_ok:
        verdict = f"🔴 ШОРТ РАЗРЕШЕН  -> {reason_short}"
        hypo = "SHORT"
    else:
        hypo = "ANY"
        verdict = f"⛔ БЛОКИРОВКА     -> {reason_long if ratio < 0 else reason_short}"

    print(f"{coin:<7} | {cvd_str:<12} | {ratio_str:<11} | {hypo:<10} | {verdict}")

print("=" * 85)
