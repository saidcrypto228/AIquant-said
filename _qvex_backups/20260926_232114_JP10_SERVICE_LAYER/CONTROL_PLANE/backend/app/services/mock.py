"""
QVEX Control Plane — Детерминированный MockTradingService для QA и UI (Section 5).
Реализует TradingService без обращения к файловой системе или внешним API.
"""
import time
from typing import Optional, List
from app.models.schemas import (
    CoreState,
    PositionState,
    ModelStatus,
    DataQuality,
    CommandResult,
    ActionType
)
from app.services.base import TradingService


class MockTradingService(TradingService):
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.trading_enabled = True
        self.system_status = "ACTIVE"
        self._init_state()

    def _init_state(self):
        if self.dry_run:
            self.equity = None
            self.free_margin = None
            self.unrealized_pnl = None
            self.btc_price = 64150.0
            self.data_quality = DataQuality(fresh=False, reason="DRY-RUN Mock Mode")
            self.positions: List[PositionState] = []
        else:
            self.equity = 12540.80
            self.free_margin = 8420.15
            self.unrealized_pnl = 312.45
            self.btc_price = 64280.0
            self.data_quality = DataQuality(fresh=True, reason=None)
            self.positions: List[PositionState] = [
                PositionState(
                    coin="SOL",
                    side="LONG",
                    size=15.5,
                    entry_px=142.20,
                    sl_px=138.50,
                    highest_px=146.80,
                    trailing_active=True,
                    breakeven_active=False,
                    ml_prob=0.74
                )
            ]

    async def get_state(self) -> CoreState:
        return CoreState(
            schema_version=1,
            timestamp=time.time(),
            system_status=self.system_status,
            trading_enabled=self.trading_enabled,
            network="TESTNET",
            account_address="0xMockUserAddress4HSwing777",
            equity=self.equity,
            free_margin=self.free_margin,
            unrealized_pnl=self.unrealized_pnl,
            btc_price=self.btc_price,
            market_regime="BULL (SLOTS: 2)",
            active_slots=len(self.positions),
            max_slots=2,
            positions=self.positions,
            model_status=ModelStatus(loaded=True, model_path="data/meta_model.json", features_count=10),
            data_quality=self.data_quality
        )

    async def pause_trading(self, actor: str = "Operator", reason: Optional[str] = None) -> CommandResult:
        self.trading_enabled = False
        self.system_status = "PAUSED"
        msg = f"Торговля приостановлена пользователем {actor}"
        if reason:
            msg += f" (Причина: {reason})"
        return CommandResult(
            action=ActionType.PAUSE,
            success=True,
            actor=actor,
            timestamp=time.time(),
            message=msg,
            metadata={"trading_enabled": False}
        )

    async def resume_trading(self, actor: str = "Operator") -> CommandResult:
        self.trading_enabled = True
        self.system_status = "ACTIVE"
        return CommandResult(
            action=ActionType.RESUME,
            success=True,
            actor=actor,
            timestamp=time.time(),
            message=f"Торговля возобновлена пользователем {actor}",
            metadata={"trading_enabled": True}
        )

    async def trigger_panic(self, actor: str = "Operator", reason: str = "Emergency flatten") -> CommandResult:
        closed_count = len(self.positions)
        self.positions = []
        self.trading_enabled = False
        self.system_status = "PAUSED"
        if not self.dry_run and self.equity is not None and self.unrealized_pnl is not None:
            self.equity += self.unrealized_pnl
            self.free_margin = self.equity
            self.unrealized_pnl = 0.0

        return CommandResult(
            action=ActionType.PANIC,
            success=True,
            actor=actor,
            timestamp=time.time(),
            message=f"Экстренная ликвидация активирована ({actor}): закрыто позиций: {closed_count}",
            metadata={"closed_positions": closed_count, "reason": reason}
        )
