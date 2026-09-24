#!/usr/bin/env python3
"""
Hyperliquid Historical Universe Ingestion.
Pulls continuous 1H OHLCV history for the 9 target assets via candleSnapshot.
Target symbols: BTC, ETH, SOL, HYPE, NEAR, SUI, XRP, DOGE, UNI.
"""

import sys
import time
import json
import requests
from pathlib import Path
import pandas as pd
import numpy as np

API_URL = "https://api.hyperliquid.xyz/info"
HEADERS = {"Content-Type": "application/json"}

TARGET_COINS = ["BTC", "ETH", "SOL", "HYPE", "NEAR", "SUI", "XRP", "DOGE", "UNI"]
INTERVAL = "1h"
BATCH_LIMIT = 5000  # Максимальный лимит биржи на один вызов
DESIRED_BARS = 12000  # ~16-18 месяцев часовой истории

SAVE_DIR = Path("data/universe")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_candles(coin: str, start_time: int, end_time: int):
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": INTERVAL,
            "startTime": start_time,
            "endTime": end_time
        }
    }
    resp = requests.post(API_URL, headers=HEADERS, json=payload, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP Error {resp.status_code}: {resp.text}")
    return resp.json()

def download_coin_history(coin: str):
    print(f"[*] Начало выгрузки {coin} (1H)...")
    end_time = int(time.time() * 1000)
    all_candles = []

    # Итеративная пагинация назад во времени
    while len(all_candles) < DESIRED_BARS:
        # Сдвиг на 5000 часов назад: 5000 * 3600 * 1000 мс
        start_time = end_time - (BATCH_LIMIT * 3600 * 1000)
        try:
            batch = fetch_candles(coin, start_time, end_time)
        except Exception as e:
            print(f"    [!] Ошибка запроса для {coin}: {e}. Повтор через 2с...")
            time.sleep(2)
            continue

        if not batch:
            print(f"    [i] Достигнут генезис/начало торгов для {coin}.")
            break

        all_candles.extend(batch)
        earliest_ts = batch[0]["t"]

        # Если биржа вернула свечи, которые не сдвинулись глубже во времени — выходим
        if earliest_ts >= end_time:
            break

        end_time = earliest_ts
        print(f"    Загружено {len(all_candles):,} баров (Самый ранний: {pd.to_datetime(earliest_ts, unit='ms', utc=True)})...")
        time.sleep(0.3)  # Соблюдение rate-limit (1200 вес/мин)

    if not all_candles:
        print(f"[-] Нет данных по {coin}!")
        return None

    # Преобразование в DataFrame
    records = []
    for c in all_candles:
        records.append({
            "timestamp": c["t"],
            "open": float(c["o"]),
            "high": float(c["h"]),
            "low": float(c["l"]),
            "close": float(c["c"]),
            "volume": float(c["v"])
        })

    df = pd.DataFrame(records)
    # Удаление дубликатов и сортировка по возрастанию времени
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    file_path = SAVE_DIR / f"{coin}_1h.csv"
    df.to_csv(file_path, index=False)
    print(f"[+] {coin} успешно сохранен: {len(df):,} баров -> {file_path}")
    return df

def main():
    print("=" * 65)
    print("  HYPERLIQUID UNIVERSE DATA INGESTION ENGINE")
    print("=" * 65)
    print(f"Корзина инструментов: {', '.join(TARGET_COINS)}")

    summary = {}
    for coin in TARGET_COINS:
        df = download_coin_history(coin)
        if df is not None:
            summary[coin] = len(df)

    print("=" * 65)
    print("ИТОГИ СБОРА ВСЕЛЕННОЙ АКТИВОВ:")
    for coin, count in summary.items():
        print(f"  - {coin:<6}: {count:,} часовых баров")
    print("=" * 65)

if __name__ == "__main__":
    main()
