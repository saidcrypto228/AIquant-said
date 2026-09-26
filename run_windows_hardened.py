import ctypes
import os
import sys
import subprocess
import time

# Флаги Windows API против спящего режима
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_AWAYMODE_REQUIRED = 0x00000040

def prevent_windows_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
        print("✔ [PowerLock] Блокировка спящего режима Windows активирована.")
    except Exception as e:
        print(f"⚠️ Не удалось заблокировать сон ОС: {e}")

def main():
    print("=" * 70)
    print("🚀 QVEX v10.7: ЗАПУСК В РЕЖИМЕ ПОВЫШЕННОЙ ОТКАЗОУСТОЙЧИВОСТИ")
    print("=" * 70)
    prevent_windows_sleep()

    # Запуск основного диспетчера системы
    cmd = [sys.executable, "-u", "run_system.py"]
    proc = subprocess.Popen(cmd)
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[*] Остановка комплекса пользователем...")
        proc.terminate()
        proc.wait()

if __name__ == "__main__":
    main()
