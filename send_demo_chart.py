#!/usr/bin/env python3
import time
import json
import urllib.request
import pandas as pd
from tg_visualizer import TelegramVisualizer

print("[*] Получаем свечи SOL с L1...")
end_ms = int(time.time() * 1000)
start_ms = end_ms - (140 * 3600 * 1000)

req_data = json.dumps({
    "type": "candleSnapshot",
    "req": {"coin": "SOL", "interval": "1h", "startTime": start_ms, "endTime": end_ms}
}).encode("utf-8")

hl_req = urllib.request.Request(
    "https://api.hyperliquid-testnet.xyz/info",
    data=req_data,
    headers={"Content-Type": "application/json", "User-Agent": "HL-Sentinel"}
)

with urllib.request.urlopen(hl_req, timeout=10) as resp:
    raw = json.loads(resp.read().decode("utf-8"))

df_raw = pd.DataFrame([{
    "timestamp": int(c["t"]),
    "datetime": pd.to_datetime(c["t"], unit="ms", utc=True),
    "open": float(c["o"]),
    "high": float(c["h"]),
    "low": float(c["l"]),
    "close": float(c["c"]),
    "volume": float(c["v"])
} for c in raw]).sort_values("timestamp")

df_4h = df_raw.set_index("datetime").resample("4h", label="right", closed="right").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna().reset_index()

c = df_4h["close"]
df_4h["ema20_4h"] = c.ewm(span=20, adjust=False).mean()
df_4h["ema50_4h"] = c.ewm(span=50, adjust=False).mean()

last_row = df_4h.iloc[-1]
entry_px = round(last_row["ema20_4h"], 2)
sl_px = round(entry_px * 0.942, 2)

print("[*] Отправляем карточку через MarkdownV2...")
ok = TelegramVisualizer.send_signal_photo_with_card(
    df_4h=df_4h,
    coin="SOL",
    sz=0.103,
    entry_px=entry_px,
    sl_px=sl_px,
    notional=12.0,
    risk_usd=3.00,
    rs_pct=2.45
)

if ok:
    print("[✓] УСПЕХ: Фото и прикрепленная MarkdownV2-карточка доставлены в Telegram!")
else:
    print("[-] Ошибка отправки карточки.")
