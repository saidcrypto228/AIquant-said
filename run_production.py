#!/usr/bin/env python3
"""
Hyperliquid Production Supervisor (v10.9-hardened).
Управляет двумя изолированными процессами:
1. hl_orderflow_ws.py: Потоковый WebSocket L1 (Trades, L2Book, activeAssetCtx Delta-OI).
2. hl_swing_bot.py: 4H торговое ядро с ML Meta-Labeling и Economic Cost Gate (Stop >= 1.20%).
"""

import subprocess
import sys
import time
import os
import signal

def run_supervisor():
    print("=" * 75)
    print("  HYPERLIQUID L1 QUANT SUITE: PRODUCTION SUPERVISOR (v10.9-hardened)")
    print("=" * 75)

    # 1. Потоковый микроструктурный сборщик (CVD + OBI_10 + Delta-OI)
    print("[*] Запуск фонового процесса Order Flow (CVD + L2Book + Delta-OI)...")
    p_ws = subprocess.Popen([sys.executable, "hl_orderflow_ws.py"])
    time.sleep(4)

    # 2. Боевое торговое ядро (4H Macro + Invariants)
    print("[*] Запуск торгового ядра с ML Meta-Labeling (hl_swing_bot.py)...")
    p_bot = subprocess.Popen([sys.executable, "hl_swing_bot.py"])

    try:
        while True:
            # Мониторинг жизнеспособности дочерних процессов
            if p_ws.poll() is not None:
                print("[-] OrderFlowWS завершился аварийно. Перезапуск через 3 сек...")
                time.sleep(3)
                p_ws = subprocess.Popen([sys.executable, "hl_orderflow_ws.py"])

            if p_bot.poll() is not None:
                print("[-] HLBot завершился аварийно. Перезапуск через 3 сек...")
                time.sleep(3)
                p_bot = subprocess.Popen([sys.executable, "hl_swing_bot.py"])

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n[!] Получен сигнал остановки (Ctrl+C). Корректное завершение процессов...")
        p_ws.terminate()
        p_bot.terminate()
        p_ws.wait()
        p_bot.wait()
        print("[✓] Все квантовые процессы безопасно остановлены.")

if __name__ == "__main__":
    run_supervisor()
