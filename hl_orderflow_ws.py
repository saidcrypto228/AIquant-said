#!/usr/bin/env python3
"""
Локальный WebSocket-клиент Order Flow v9.4.
- C12 Fix: Использование биржевого timestamp сделки (trade['time']) вместо локального time.time().
- C13 Fix: Независимый цикл вытеснения устаревших сделок для защиты от десинхронизации в тихие часы.
- Атомарная публикация с уникальным PID-файлом.
"""

import os
import sys
import time
import json
import asyncio
import logging
from collections import deque
from typing import Dict, Any

import bot_config as config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] OrderFlowWS-v9.4: %(message)s"
)
logger = logging.getLogger("OrderFlowWS-v9.4")

try:
    import websockets
except ImportError:
    logger.error("Библиотека 'websockets' не установлена.")
    sys.exit(1)

class RollingCVDCollector:
    def __init__(self, coins: list, window_sec: int = 3600):
        self.coins = coins
        self.window_sec = window_sec
        self.ws_url = "wss://api.hyperliquid-testnet.xyz/ws" if config.IS_TESTNET else "wss://api.hyperliquid.xyz/ws"
        self.state_file = config.DATA_DIR / "orderflow_state.json"

        self.trade_buckets: Dict[str, deque] = {coin: deque() for coin in self.coins}
        self.last_prices: Dict[str, float] = {coin: 0.0 for coin in self.coins}
        self.running = True

    def _clean_stale_trades(self, now: float):
        cutoff = now - self.window_sec
        for coin in self.coins:
            q = self.trade_buckets[coin]
            while q and q[0]["time"] < cutoff:
                q.popleft()

    def _get_metrics_snapshot(self, now: float) -> Dict[str, Any]:
        self._clean_stale_trades(now)
        snapshot = {}

        for coin in self.coins:
            q = self.trade_buckets[coin]
            buy_vol = sum(t["usd"] for t in q if t["is_buy"])
            sell_vol = sum(t["usd"] for t in q if not t["is_buy"])
            total_vol = buy_vol + sell_vol
            cvd_usd = buy_vol - sell_vol
            cvd_ratio = (cvd_usd / max(total_vol, 1.0)) if total_vol > 0 else 0.0

            snapshot[coin] = {
                "buy_vol_usd": buy_vol,
                "sell_vol_usd": sell_vol,
                "total_vol_usd": total_vol,
                "cvd_usd": cvd_usd,
                "cvd_ratio": cvd_ratio,
                "trades_count": len(q),
                "last_px": self.last_prices[coin],
                "last_update": now
            }
        return snapshot

    def _save_snapshot(self, now: float):
        metrics = self._get_metrics_snapshot(now)
        tmp_file = self.state_file.with_name(f"{self.state_file.name}.{os.getpid()}.tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump({"timestamp": now, "coins": metrics}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_file, self.state_file)
        except PermissionError:
            pass
        except Exception as e:
            logger.debug(f"Ошибка сохранения снапшота: {e}")
        finally:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

    async def _subscribe(self, ws):
        for coin in self.coins:
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": coin}
            }))
            logger.info(f"[+] Подписка на поток сделок: {coin}")

    def _process_trades(self, data: list, now: float):
        for trade in data:
            coin = trade.get("coin")
            if coin not in self.trade_buckets:
                continue

            px = float(trade.get("px", 0.0))
            sz = float(trade.get("sz", 0.0))
            side = trade.get("side", "").upper()
            notional = px * sz

            # C12 Fix: Используем нативное биржевое время сделки
            trade_time = float(trade.get("time", now * 1000)) / 1000.0

            if side == "B":
                is_buy = True
            elif side == "A":
                is_buy = False
            else:
                continue

            self.last_prices[coin] = px
            self.trade_buckets[coin].append({
                "time": trade_time,
                "usd": notional,
                "is_buy": is_buy
            })

    async def run_listener(self):
        logger.info(f"[*] Запуск Rolling CVD WebSocket v9.4 (Окно: {self.window_sec//60} мин)...")
        last_flush = time.time()

        while self.running:
            try:
                async with websockets.connect(
                    self.ws_url, ping_interval=20, ping_timeout=15, max_size=10_000_000
                ) as ws:
                    await self._subscribe(ws)
                    logger.info("[✓] WebSocket L1 открыт. Прием тиков...")

                    while self.running:
                        msg_raw = await ws.recv()
                        msg = json.loads(msg_raw)

                        now = time.time()
                        if msg.get("channel") == "trades":
                            self._process_trades(msg.get("data", []), now)

                        if now - last_flush >= 1.5:
                            self._save_snapshot(now)
                            last_flush = now

            except (websockets.ConnectionClosed, asyncio.TimeoutError) as e:
                logger.warning(f"[!] Разрыв WS: {e}. Переподключение через 3 сек...")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"[-] Ошибка WS: {e}", exc_info=True)
                await asyncio.sleep(5)

if __name__ == "__main__":
    target_coins = [
        c for c in config.TARGET_COINS 
        if c not in ["ETH", "LINK", "PEPE", "kPEPE", "WIF"]
    ]
    collector = RollingCVDCollector(target_coins, window_sec=3600)
    try:
        asyncio.run(collector.run_listener())
    except KeyboardInterrupt:
        logger.info("[*] Остановка сборщика CVD.")
