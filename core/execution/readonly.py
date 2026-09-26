"""
QVEX v10.7 — Read-Only модуль интеграции с Hyperliquid.
"""
from __future__ import annotations
import time
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
import httpx

logger = logging.getLogger("qvex.hl_readonly")
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

@dataclass
class _CacheEntry:
    value: Any
    fetched_at: float

class HyperliquidReadOnlyClient:
    def __init__(self, account_address: Optional[str] = None, timeout: float = 10.0):
        self.account_address = (account_address or ZERO_ADDRESS).lower()
        self.dry_run = self.account_address == ZERO_ADDRESS
        self._client = httpx.AsyncClient(timeout=timeout)
        self._cache: Dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def aclose(self):
        await self._client.aclose()

    def _cache_get(self, key: str, ttl: float):
        entry = self._cache.get(key)
        if entry and (time.monotonic() - entry.fetched_at) < ttl:
            return entry.value
        return None

    def _cache_set(self, key: str, value: Any):
        self._cache[key] = _CacheEntry(value=value, fetched_at=time.monotonic())

    async def _post_info(self, payload: dict) -> Any:
        resp = await self._client.post(HL_INFO_URL, json=payload)
        resp.raise_for_status()
        return resp.json()

    async def get_clearinghouse_state(self) -> dict:
        if self.dry_run:
            return {
                "marginSummary": {"accountValue": "0", "totalMarginUsed": "0"},
                "assetPositions": [],
                "withdrawable": "0"
            }
        cache_key = f"chs:{self.account_address}"
        cached = self._cache_get(cache_key, 10.0)
        if cached is not None:
            return cached
        async with self._lock:
            try:
                data = await self._post_info({"type": "clearinghouseState", "user": self.account_address})
                self._cache_set(cache_key, data)
                return data
            except Exception as e:
                logger.error("Clearinghouse fetch error: %s", e)
                return {"marginSummary": {"accountValue": "0", "totalMarginUsed": "0"}, "assetPositions": [], "withdrawable": "0"}

    async def get_candles(self, coin: str = "BTC", interval: str = "4h", lookback_hours: int = 72) -> List[dict]:
        cache_key = f"candles:{coin}:{interval}:{lookback_hours}"
        cached = self._cache_get(cache_key, 45.0)
        if cached is not None:
            return cached
        async with self._lock:
            now_ms = int(time.time() * 1000)
            start_ms = now_ms - lookback_hours * 3600 * 1000
            try:
                data = await self._post_info({
                    "type": "candleSnapshot",
                    "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": now_ms}
                })
                normalized = [
                    {"time": c["t"] // 1000, "open": float(c["o"]), "high": float(c["h"]), "low": float(c["l"]), "close": float(c["c"]), "volume": float(c["v"])}
                    for c in data
                ]
                self._cache_set(cache_key, normalized)
                return normalized
            except Exception as e:
                logger.error("Candles fetch error: %s", e)
                return []

def parse_positions_from_clearinghouse(chs: dict) -> Dict[str, dict]:
    out = {}
    for ap in chs.get("assetPositions", []):
        pos = ap.get("position", {})
        sym = pos.get("coin")
        szi = float(pos.get("szi", 0))
        if sym and szi != 0:
            out[sym] = {
                "symbol": sym, "side": "LONG" if szi > 0 else "SHORT",
                "size": abs(szi), "entry_price": float(pos.get("entryPx", 0)),
                "unrealized_pnl": float(pos.get("unrealizedPnl", 0))
            }
    return out
