"""
=============================================================================
                    QVEX v10.7 — CORE RUNNER (AUTONOMOUS)
=============================================================================
Чистая точка запуска автономного торгового ядра QVEX на Hyperliquid L1.
Запуск:
    python qvex.py
=============================================================================
"""
import os
import sys
import signal
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [QVEX-CORE-RUNNER]: %(message)s"
)
logger = logging.getLogger("QVEX.Runner")

def print_banner():
    is_testnet = os.getenv("TESTNET", "True").lower() == "true"
    net_str = "TESTNET (Песочница)" if is_testnet else "MAINNET (Реальный счет)"

    print("=" * 72)
    print("        🚀 QVEX v10.7: АВТОНОМНЫЙ ЗАПУСК ТОРГОВОГО ЯДРА (БЕЗ БОТА)     ")
    print("=" * 72)
    print(f"• Сеть:             {net_str}")
    print(f"• Режим:            Изолированное квантовое ядро")
    print(f"• Архитектура:      4H Swing + Native L1 Stops + Atomic POSIX IPC")
    print("=" * 72)
    print("Для безопасной остановки нажмите Ctrl + C\n")

def main():
    print_banner()

    core_script = ROOT_DIR / "hl_swing_bot.py"
    if not core_script.exists():
        logger.critical(f"Файл торгового ядра не найден: {core_script}")
        sys.exit(1)

    logger.info("Запуск автономного торгового процесса (hl_swing_bot.py)...")
    core_proc = subprocess.Popen([sys.executable, str(core_script)], cwd=str(ROOT_DIR))

    def shutdown(signum, frame):
        logger.info("Получен сигнал завершения. Остановка торгового ядра...")
        if core_proc.poll() is None:
            core_proc.terminate()
            try:
                core_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                core_proc.kill()
        logger.info("✔ Торговое ядро успешно остановлено.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        core_proc.wait()
    except KeyboardInterrupt:
        shutdown(None, None)

if __name__ == "__main__":
    main()
