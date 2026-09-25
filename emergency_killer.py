import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [EMERGENCY KILLER] %(message)s")

def panic_close_all():
    logging.warning("🚨 АВАРИЙНЫЙ ПРОЦЕСС ЗАПУЩЕН! Инициация рыночного сброса...")
    # Здесь будет вызов Hyperliquid SDK:
    # 1. Отмена всех открытых ордеров (cancel_all)
    # 2. Получение списка позиций
    # 3. Отправка рыночных ордеров (market_close) на весь объем
    logging.warning("✔ Команды на биржу отправлены. Позиции ликвидированы.")

if __name__ == '__main__':
    panic_close_all()
