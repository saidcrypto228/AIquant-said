"""
QVEX v10.7 — Проверка рассинхронизации системного времени (HyperBFT Drift Check).
Проверяет расхождение локальных часов относительно биржевых серверов Hyperliquid.
Лимит консенсуса: строго < 1000 мс.
"""
import time
import requests

def verify_hyperliquid_drift():
    print("=" * 60)
    print("⏱️ ПРОВЕРКА ДРЕЙФА СИСТЕМНОГО ВРЕМЕНИ ДЛЯ HYPERLIQUID L1")
    print("=" * 60)

    t_start = time.time()
    try:
        response = requests.post(
            "https://api.hyperliquid.xyz/info",
            json={"type": "meta"},
            timeout=5
        )
        t_recv = time.time()

        # Оценка сетевой задержки (Round-Trip Time)
        rtt_ms = (t_recv - t_start) * 1000

        # Серверный заголовок даты
        date_header = response.headers.get("Date")
        if not date_header:
            print("⚠️ Заголовок Date отсутствует в ответе ноды.")
            return

        from email.utils import parsedate_to_datetime
        server_dt = parsedate_to_datetime(date_header)
        server_ts = server_dt.timestamp()
        local_ts = t_recv

        drift_ms = abs(local_ts - server_ts) * 1000
        print(f"• Сетевая задержка (RTT): {rtt_ms:.1f} мс")
        print(f"• Расхождение времени:    {drift_ms:.1f} мс")

        if drift_ms < 1000:
            print("✔ [СТАТУС: В НОРМЕ] Дрейф часов соответствует консенсусу HyperBFT (< 1000 мс).")
        else:
            print("❌ [СТАТУС: РИСК] Расхождение превышает 1000 мс! Необходима синхронизация chrony.")

    except Exception as e:
        print(f"Ошибка проверки времени: {e}")
    print("=" * 60)

if __name__ == "__main__":
    verify_hyperliquid_drift()
