"""
QVEX v10.7 — Декларативные схемы состояния торгового ядра и телеметрии.
Исключительно Pydantic-модели без прямого дискового ввода/вывода.
"""
from typing import Dict, Optional
from pydantic import BaseModel, Field

class MarketRegime(BaseModel):
    status: str = Field(..., description="Рыночный режим: STRONG_BULL, BULL, CHOP, BEAR, STRONG_BEAR")
    btc_price: float = Field(..., description="Текущая индикативная цена BTC")
    slope: float = Field(default=0.0, description="Наклон скользящей EMA")

class AccountInfo(BaseModel):
    equity: float = Field(..., description="Текущий баланс депозита")
    free_margin: float = Field(..., description="Свободная маржа")
    total_balance: float = Field(..., description="Общий баланс счета")
    max_slots: int = Field(default=2, description="Максимум одновременно торгуемых пар")

class Position(BaseModel):
    symbol: str = Field(..., description="Тикер актива (BTC, ETH, SOL)")
    side: str = Field(..., description="Направление: LONG / SHORT")
    size: float = Field(..., description="Размер позиции в базовой валюте")
    entry_price: float = Field(..., description="Средневзвешенная цена входа")
    current_price: float = Field(..., description="Текущая рыночная цена")
    unrealized_pnl: float = Field(default=0.0, description="Нереализованный PnL в USD")
    chandelier_stop: float = Field(..., description="Актуальный уровень биржевого трейлинг-стопа")

class StateSnapshot(BaseModel):
    total_equity: float = Field(..., description="Совокупная ликвидационная стоимость")
    free_margin: float = Field(..., description="Маржинальный резерв")
    btc_price: float = Field(..., description="Цена бенчмарка BTC")
    account_address: str = Field(..., description="Адрес аккаунта Hyperliquid")
    market_regime: MarketRegime = Field(..., description="Метрики рыночного режима")
    account: AccountInfo = Field(..., description="Параметры счета")
    positions: Dict[str, Position] = Field(default_factory=dict, description="Словарь открытых позиций")

# --- ФУНКЦИИ ОБРАТНОЙ СОВМЕСТИМОСТИ (BRIDGES TO STATE_IPC) ---
def write_state_atomic(state: StateSnapshot, filepath: str = "data/bot_state.json") -> None:
    from state_ipc import PosixAtomicStateManager
    mgr = PosixAtomicStateManager(filepath, StateSnapshot)
    mgr.write_atomic_state(state)

def read_state_safe(filepath: str = "data/bot_state.json") -> StateSnapshot:
    from state_ipc import PosixAtomicStateManager
    mgr = PosixAtomicStateManager(filepath, StateSnapshot)
    return mgr.read_atomic_state()
