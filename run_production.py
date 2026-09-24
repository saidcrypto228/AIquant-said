#!/usr/bin/env python3
"""
Институциональный супервизор (Production Supervisor).
Одновременно координирует:
1. Фоновый WebSocket-клиент Order Flow (hl_orderflow_ws.py).
2. Основной торговый контур с ML Meta-Labeling (hl_swing_bot.py).
"""

import sys
import time
import subprocess
from pathlib import Path

print("=" * 75)
print("  HYPERLIQUID L1 QUANT SUITE: PRODUCTION SUPERVISOR")
print("=" * 75)

python_exe = sys.executable

# 1. Запуск WebSocket Order Flow
print("[*] Запуск фонового процесса Order Flow (CVD WebSocket)...")
p_ws = subprocess.Popen([python_exe, "hl_orderflow_ws.py"])
time.sleep(3)

# 2. Запуск основного торгового ядра
print("[*] Запуск торгового ядра с ML Meta-Labeling (hl_swing_bot.py)...")
p_bot = subprocess.Popen([python_exe, "hl_swing_bot.py"])

try:
    while True:
        if p_ws.poll() is not None:
            print("[!] Внимание: процесс Order Flow завершился. Перезапуск...")
            p_ws = subprocess.Popen([python_exe, "hl_orderflow_ws.py"])
        if p_bot.poll() is not None:
            print("[!] Внимание: торговый бот завершился. Перезапуск...")
            p_bot = subprocess.Popen([python_exe, "hl_swing_bot.py"])
        time.sleep(5)
except KeyboardInterrupt:
    print("\n[*] Завершение работы квантового комплекса...")
    p_bot.terminate()
    p_ws.terminate()
    p_bot.wait()
    p_ws.wait()
    print("[✓] Все процессы остановлены.")
