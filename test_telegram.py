#!/usr/bin/env python3
"""
Скрипт проверки доставки сообщений через Telegram Sentinel.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
from pathlib import Path

# Загрузка .env
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

token = os.getenv("TELEGRAM_BOT_TOKEN", "8160908398:AAFEBPHMBd5rVY0azyDazZoOnumZhqd5CDo").strip()
chat_id = os.getenv("TELEGRAM_CHAT_ID", "7001461641").strip()

print("=" * 60)
print("  ПРОВЕРКА СВЯЗИ С TELEGRAM SENTINEL")
print("=" * 60)
print(f"Токен бота : {token[:12]}...{token[-6:]}")
print(f"Chat ID    : {chat_id}")

message = """🚀 *Hyperliquid 4H Swing Bot: Связь Установлена!*

🟢 *Статус:* Sentinel в сети и готов к мониторингу
🛡 *Защита:*
  • On-chain Dead-Man's Switch: `АКТИВЕН`
  • Anti-Euphoria Funding Gate: `АКТИВЕН` (лимит 35% APR)
  • Exchange-Side Native SL: `ГОТОВ`
  • FSM & State Reconciliation: `СИНХРОНИЗИРОВАНО`

_Телеметрия подключена. Все торговые события будут приходить сюда._"""

try:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"User-Agent": "HL-Swing-Bot"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status == 200:
            print("\n[✓] УСПЕХ: Проверочное сообщение доставлено в Telegram!")
        else:
            print(f"\n[-] Сервер вернул HTTP-код: {resp.status}")
except Exception as e:
    print(f"\n[-] Ошибка отправки: {e}")
    print("    Если вы еще не нажали /start в боте @AiQuantVector_bot — сделайте это.")

print("=" * 60)
