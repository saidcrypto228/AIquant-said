#!/usr/bin/env python3
import time
import json
import asyncio
from pathlib import Path
import bot_config as config
from hl_orderflow_ws import HyperliquidOrderFlowCollector

print("=" * 82)
print("  ТЕСТ ОНЧЕЙН ORDER FLOW: ЖИВОЙ ПРИЕМ ТИКОВ С HYPERLIQUID L1")
print("=" * 82)

target_coins = [
    c for c in config.TARGET_COINS 
    if c not in ["ETH", "LINK", "PEPE", "kPEPE", "WIF"]
]

collector = HyperliquidOrderFlowCollector(target_coins)

async def test_session():
    # Запуск сборщика в фоне
    task = asyncio.create_task(collector.run_listener())

    print("[*] Сбор тиков в реальном времени (сессия 25 секунд)...")
    for sec in range(25, 0, -5):
        print(f"  > Осталось {sec} сек...")
        await asyncio.sleep(5)

    collector.running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

asyncio.run(test_session())

print("\n" + "=" * 82)
print("  СВОДКА МИКРОСТРУКТУРЫ: СРЕЗ КУМУЛЯТИВНОЙ ДЕЛЬТЫ (CVD)")
print("=" * 82)

header_asset = "АКТИВ"
header_px = "ЦЕНА"
header_cnt = "СДЕЛОК"
header_buy = "ПОКУПКИ ($)"
header_sell = "ПРОДАЖИ ($)"
header_cvd = "CVD ($)"
header_ratio = "CVD RATIO"

print(f"{header_asset:<7} | {header_px:<9} | {header_cnt:<7} | {header_buy:<13} | {header_sell:<13} | {header_cvd:<13} | {header_ratio}")
print("-" * 82)

for coin, m in collector.metrics.items():
    ratio_pct = m["cvd_ratio"] * 100
    ratio_str = f"{ratio_pct:+6.1f}%"
    cvd_str = f"${m['cvd_usd']:+,.1f}"
    buy_str = f"${m['buy_vol_usd']:,.1f}"
    sell_str = f"${m['sell_vol_usd']:,.1f}"
    px_val = m["last_px"]
    px_str = f"${px_val:<8.2f}"

    print(f"{coin:<7} | {px_str:<9} | {m['trades_count']:<7} | {buy_str:<13} | {sell_str:<13} | {cvd_str:<13} | {ratio_str}")

print("-" * 82)
print("✓ Данные сохранены в data/orderflow_state.json")
print("=" * 82)
