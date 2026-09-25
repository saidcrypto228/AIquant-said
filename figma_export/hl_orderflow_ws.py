#!/usr/bin/env python3
"""
Hyperliquid Order Flow & Microstructure Collector (v10.8 - CVD + OBI_10).
- Потоковый сбор тиков trades для 11 инструментов L1.
- Потоковый сбор стакана L2 (l2Book) и расчет взвешенного дисбаланса OBI_10.
- Расчет скользящего 60-мин CVD (Cumulative Volume Delta) и last_px.
- Встроенный фоновый JSON Keep-Alive пинг (каждые 15 сек).
- Атомарная запись data/orderflow_state.json каждые 1.0 сек.
"""

import asyncio
import json
import logging
import sys
import time
import os
from collections import deque
from pathlib import Path
from typing import Dict, Any

import websockets
import numpy as np

import bot_config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] OrderFlowWS-v10.8: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("OrderFlowWS-v10.8")

CLEAN_TARGETS = [c for c in config.TARGET_COINS if c not in ["ETH", "LINK", "PEPE", "kPEPE", "WIF"]]
WS_URL = "wss://api.hyperliquid-testnet.xyz/ws" if config.IS_TESTNET else "wss://api.hyperliquid.xyz/ws"
WINDOW_SEC = 3600

class MicrostructureCollector:
    def __init__(self):
        self.trades_history: Dict[str, deque] = {coin: deque() for coin in CLEAN_TARGETS}
        self.last_prices: Dict[str, float] = {coin: 0.0 for coin in CLEAN_TARGETS}
        self.book_imbalance: Dict[str, float] = {coin: 0.0 for coin in CLEAN_TARGETS}
        self.state_file = config.DATA_DIR / "orderflow_state.json"
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.running = True

    async def _send_keepalive_pings(self, ws):
        """Пинг L1 сервера каждые 15 сек для опережения таймаутов прокси/Cloudflare."""
        while self.running:
            try:
                await asyncio.sleep(15)
                if ws.open:
                    await ws.send(json.dumps({"method": "ping"}))
            except asyncio.CancelledError:
                break
            except Exception:
                break

    def _compute_obi(self, bids: list, asks: list, depth: int = 10) -> float:
        """
        Расчет взвешенного дисбаланса книги заявок (Orderbook Imbalance OBI_10).
        Веса уровней затухают как w_i = 1 / (i + 1).
        """
        if not bids or not asks:
            return 0.0

        n = min(depth, len(bids), len(asks))
        if n == 0:
            return 0.0

        bid_weighted = sum(float(bids[i]["sz"]) / (i + 1) for i in range(n))
        ask_weighted = sum(float(asks[i]["sz"]) / (i + 1) for i in range(n))
        total = bid_weighted + ask_weighted

        if total <= 1e-9:
            return 0.0

        return float(np.clip((bid_weighted - ask_weighted) / total, -1.0, 1.0))

    async def _save_state_loop(self):
        while self.running:
            try:
                now = time.time()
                coins_payload = {}
                cutoff = now - WINDOW_SEC

                for coin in CLEAN_TARGETS:
                    q = self.trades_history[coin]
                    while q and q[0][0] < cutoff:
                        q.popleft()

                    buy_vol = sum(vol for _, side, vol, _ in q if side == "B")
                    sell_vol = sum(vol for _, side, vol, _ in q if side == "A")
                    total_vol = buy_vol + sell_vol
                    cvd_usd = buy_vol - sell_vol
                    cvd_ratio = (cvd_usd / total_vol) if total_vol > 0 else 0.0

                    coins_payload[coin] = {
                        "cvd_usd": round(cvd_usd, 2),
                        "cvd_ratio": round(cvd_ratio, 4),
                        "total_volume_usd": round(total_vol, 2),
                        "obi_10": round(self.book_imbalance[coin], 4),
                        "last_px": self.last_prices[coin],
                        "ticks_in_window": len(q)
                    }

                payload = {
                    "timestamp": now,
                    "coins": coins_payload
                }

                tmp = self.state_file.with_suffix(f".{os.getpid()}.tmp")
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2)
                os.replace(tmp, self.state_file)

            except Exception as e:
                logger.error(f"[-] Ошибка сохранения orderflow_state: {e}")

            await asyncio.sleep(1.0)

    async def run(self):
        logger.info(f"[*] Запуск Microstructure Collector v10.8 (CVD + OBI_10 | Целей: {len(CLEAN_TARGETS)})...")
        asyncio.create_task(self._save_state_loop())

        while self.running:
            try:
                async with websockets.connect(
                    WS_URL,
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=5,
                    max_size=20_000_000
                ) as ws:
                    ping_task = asyncio.create_task(self._send_keepalive_pings(ws))

                    for coin in CLEAN_TARGETS:
                        # 1. Подписка на сделки (trades)
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "trades", "coin": coin}}))
                        # 2. Подписка на стакан L2 (l2Book)
                        await ws.send(json.dumps({"method": "subscribe", "subscription": {"type": "l2Book", "coin": coin}}))

                    logger.info("[✓] WebSocket L1 открыт (Trades + L2Book). Прием микроструктуры...")

                    async for message in ws:
                        data = json.loads(message)
                        channel = data.get("channel")

                        if data.get("response") == "pong":
                            continue

                        # Обработка сделок
                        if channel == "trades":
                            trades = data.get("data", [])
                            for t in trades:
                                coin = t.get("coin")
                                if coin in self.trades_history:
                                    px = float(t.get("px", 0.0))
                                    sz = float(t.get("sz", 0.0))
                                    side = t.get("side")
                                    self.last_prices[coin] = px
                                    self.trades_history[coin].append((time.time(), side, px * sz, px))

                        # Обработка стакана L2 для OBI_10
                        elif channel == "l2Book":
                            book_data = data.get("data", {})
                            coin = book_data.get("coin")
                            if coin in self.book_imbalance:
                                levels = book_data.get("levels", [[], []])
                                bids = levels[0] if len(levels) > 0 else []
                                asks = levels[1] if len(levels) > 1 else []
                                self.book_imbalance[coin] = self._compute_obi(bids, asks, depth=10)

                    ping_task.cancel()
                    logger.info("[-] Сессия WS завершена удаленным сервером. Переподключение...")

            except (websockets.exceptions.ConnectionClosedError, websockets.exceptions.ConnectionClosedOK) as e:
                logger.warning(f"[!] Закрытие WS ({e}). Чистое переподключение через 2 сек...")
                await asyncio.sleep(2.0)
            except Exception as e:
                logger.error(f"[!] Сбой WS: {e}. Переподключение через 5 сек...")
                await asyncio.sleep(5.0)

if __name__ == "__main__":
    collector = MicrostructureCollector()
    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        logger.info("[-] Остановка по сигналу.")
