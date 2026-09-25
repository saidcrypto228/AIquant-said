from __future__ import annotations
import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("TradingStateBridge")

@dataclass(slots=True)
class PositionSchema:
    symbol: str
    side: str
    entry_price: float
    current_price: float
    pnl_usd: float
    pnl_pct: float
    sl_price: float
    sl_distance_pct: float
    is_breakeven: bool

@dataclass(slots=True)
class AccountSummarySchema:
    total_balance: float
    equity: float
    free_margin: float
    occupied_slots: str
    active_triggers: List[str] = field(default_factory=list)

@dataclass(slots=True)
class MarketRegimeSchema:
    status: str
    slope: float
    btc_price: float
    storm_filter_active: bool

class TradingStateBridge:
    def __init__(self, state_file_path: Optional[Union[str, Path]] = None, stale_threshold_sec: float = 60.0) -> None:
        self.state_file_path = Path(state_file_path) if state_file_path else Path("data/bot_state.json")
        self.stale_threshold_sec = stale_threshold_sec
        self._lock = asyncio.Lock()
        self._last_valid_state: Dict[str, Any] = {}

    async def get_account_summary(self) -> Dict[str, Any]:
        state = await self._resolve_state()
        if not state:
            return {"total_balance": 1000.0, "equity": 1000.0, "free_margin": 1000.0, "occupied_slots": "0/2", "active_triggers": []}
        acc = state.get("account", {})
        total_balance = float(acc.get("total_balance", 1000.0))
        equity = float(acc.get("equity", total_balance))
        free_margin = float(acc.get("free_margin", total_balance))
        max_slots = acc.get("max_slots", 2)
        used_slots = len(state.get("positions", []))
        return asdict(AccountSummarySchema(
            total_balance=round(total_balance, 2),
            equity=round(equity, 2),
            free_margin=round(free_margin, 2),
            occupied_slots=f"{used_slots}/{max_slots}",
            active_triggers=state.get("active_triggers", [])
        ))

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        state = await self._resolve_state()
        if not state:
            return []
        parsed = []
        raw_p = state.get("positions", {})
        pos_iterable = raw_p.values() if isinstance(raw_p, dict) else (raw_p if isinstance(raw_p, list) else [])
        for pos in pos_iterable:
            entry = float(pos.get("entry_price", 0.0))
            current = float(pos.get("current_price", entry))
            sl = float(pos.get("sl_price", 0.0))
            sl_dist = round(abs(current - sl) / current * 100, 2) if current > 0 and sl > 0 else 0.0
            pnl_usd = float(pos.get("pnl_usd", 0.0))
            pnl_pct = float(pos.get("pnl_pct", 0.0))
            if pnl_pct == 0.0 and entry > 0:
                pnl_pct = round(((current - entry) / entry) * 100, 2)
            parsed.append(asdict(PositionSchema(
                symbol=pos.get("symbol", "UNKNOWN"),
                side=str(pos.get("side", "LONG")),
                entry_price=entry,
                current_price=current,
                pnl_usd=round(pnl_usd, 2),
                pnl_pct=round(pnl_pct, 2),
                sl_price=sl,
                sl_distance_pct=sl_dist,
                is_breakeven=bool(pos.get("is_breakeven", False))
            )))
        return parsed

    async def get_market_regime(self) -> Dict[str, Any]:
        state = await self._resolve_state()
        if not state:
            return {"status": "BULL", "slope": 0.15, "btc_price": 84933.0, "storm_filter_active": False}
        reg = state.get("market_regime", {})
        return asdict(MarketRegimeSchema(
            status=str(reg.get("status", "BULL")).upper(),
            slope=round(float(reg.get("slope", 0.15)), 4),
            btc_price=round(float(reg.get("btc_price", 84933.0)), 2),
            storm_filter_active=bool(reg.get("storm_filter_active", False))
        ))

    async def _resolve_state(self) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if not self.state_file_path.exists():
                return self._last_valid_state or None
            return await asyncio.to_thread(self._sync_read)

    def _sync_read(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.state_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._last_valid_state = data
            return data
        except Exception:
            return self._last_valid_state or None

    @staticmethod
    def dump_state_atomically(state_dict: Dict[str, Any], target_path: Union[str, Path]) -> None:
        path = Path(target_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False, prefix=".state_tmp_") as tf:
            json.dump(state_dict, tf, ensure_ascii=False, indent=2)
            tf.flush()
            os.fsync(tf.fileno())
            temp_name = tf.name
        try:
            os.replace(temp_name, path)
        except Exception:
            if os.path.exists(temp_name): os.remove(temp_name)
