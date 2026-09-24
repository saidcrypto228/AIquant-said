#!/usr/bin/env python3
"""
Binance Vision Deep History Downloader (2023-2024).
Downloads monthly 1h klines for USD-M Futures directly into standardized CSVs.
No API keys required.
"""

import io
import sys
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
import pandas as pd

SAVE_DIR = Path("data/history_deep")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SYMBOLS = ["BTCUSDT", "SOLUSDT", "NEARUSDT", "SUIUSDT", "ETHUSDT"]
YEARS = [2023, 2024]
MONTHS = list(range(1, 13))

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

KLINE_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

def download_month_kline(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{symbol}/1h/{symbol}-1h-{year}-{month:02d}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            zip_bytes = resp.read()
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
                csv_names = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_names:
                    return None
                with z.open(csv_names[0]) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
                    # Если первая строка текстовый заголовок, убираем
                    if not str(df.iloc[0, 0]).isdigit():
                        df = df.iloc[1:].reset_index(drop=True)
                    df.columns = KLINE_COLUMNS[:len(df.columns)]
                    return df
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        print(f"    [!] Ошибка HTTP {e.code} при скачивании {symbol} {year}-{month:02d}")
        return None
    except Exception as e:
        print(f"    [!] Сбой сети на {symbol} {year}-{month:02d}: {e}")
        return None

def main():
    print("=" * 65)
    print("  ЗАГРУЗКА ИСТОРИИ 2023-2024 ИЗ BINANCE VISION (USD-M FUTURES)")
    print("=" * 65)

    for symbol in TARGET_SYMBOLS:
        coin = symbol.replace("USDT", "")
        out_file = SAVE_DIR / f"{coin}_1h.csv"

        print(f"\n[*] Сбор данных для {coin} ({symbol})...")
        monthly_dfs = []

        for year in YEARS:
            for month in MONTHS:
                df_month = download_month_kline(symbol, year, month)
                if df_month is not None and not df_month.empty:
                    monthly_dfs.append(df_month)
                    sys.stdout.write(f"\r    [+] Загружен: {year}-{month:02d} ({len(df_month)} баров)")
                    sys.stdout.flush()

        if not monthly_dfs:
            print(f"\n[-] Не удалось загрузить данные для {symbol}")
            continue

        full_df = pd.concat(monthly_dfs, ignore_index=True)

        # Стандартизация типов данных
        full_df["timestamp"] = full_df["timestamp"].astype(int)
        for col in ["open", "high", "low", "close", "volume"]:
            full_df[col] = full_df[col].astype(float)

        full_df["datetime"] = pd.to_datetime(full_df["timestamp"], unit="ms", utc=True)
        full_df = full_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Оставляем только нужные столбцы
        clean_df = full_df[["timestamp", "datetime", "open", "high", "low", "close", "volume"]].copy()
        clean_df.to_csv(out_file, index=False)

        start_date = clean_df['datetime'].iloc[0].strftime('%Y-%m-%d')
        end_date = clean_df['datetime'].iloc[-1].strftime('%Y-%m-%d')
        print(f"\n    [✓] Итого {coin}: {len(clean_df):,} баров | {start_date} -> {end_date}")
        print(f"        Сохранено в: {out_file}")

    print("\n" + "=" * 65)
    print("✓ Загрузка глубокой истории успешно завершена!")
    print("=" * 65)

if __name__ == "__main__":
    main()
