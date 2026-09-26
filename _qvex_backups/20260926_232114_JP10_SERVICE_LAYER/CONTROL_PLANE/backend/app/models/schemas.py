"""
QVEX Control Plane — Pydantic v2 контракты данных и команд (Audit Sections 8 & 9).
Строгая изоляция, поддержка null-эквити для DRY-RUN и аудит-трейла команд.
"""
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    PANIC = "PANIC"


class PositionState(BaseModel):
    coin: str = Field(..., description="Тикер актива")
    side: str = Field(..., description="LONG / SHORT")
    size: float = Field(..., description="Размер позиции")
    entry_px: float = Field(..., description="Цена входа")
    sl_px: Optional[float] = Field(default=None, description="Биржевой стоп-лосс")
    highest_px: Optional[float] = Field(default=None)
    lowest_px: Optional[float] = Field(default=None)
    trailing_active: bool = Field(default=False)
    breakeven_active: bool = Field(default=False)
    ml_prob: Optional[float] = Field(default=None)


class ModelStatus(BaseModel):
    loaded: bool = Field(default=False)
    model_path: Optional[str] = None
    features_count: int = Field(default=0)


class DataQuality(BaseModel):
    fresh: bool = Field(..., description="Флаг актуальности данных")
    reason: Optional[str] = Field(default=None, description="Причина неполноты данных")


class CoreState(BaseModel):
    schema_version: int = Field(default=1)
    timestamp: float = Field(...)
    system_status: str = Field(default="ACTIVE", description="ACTIVE / PAUSED / ERROR")
    trading_enabled: bool = Field(...)
    network: str = Field(default="TESTNET")
    account_address: Optional[str] = Field(default=None)
    equity: Optional[float] = Field(default=None)
    free_margin: Optional[float] = Field(default=None)
    unrealized_pnl: Optional[float] = Field(default=None)
    btc_price: Optional[float] = Field(default=None)
    market_regime: Optional[str] = Field(default=None)
    active_slots: int = Field(default=0)
    max_slots: int = Field(default=2)
    positions: List[PositionState] = Field(default_factory=list)
    model_status: Optional[ModelStatus] = Field(default=None)
    data_quality: DataQuality = Field(...)


class CommandResult(BaseModel):
    action: ActionType
    success: bool
    actor: str = Field(..., description="Инициатор команды (Telegram ID / Operator)")
    timestamp: float = Field(...)
    message: str = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)
