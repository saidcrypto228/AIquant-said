"""
QVEX Control Plane — Абстрактный сервисный контракт TradingService (Section 5).
Обеспечивает независимость UI (Telegram Bot & WebApp) от транспортного слоя.
"""
from abc import ABC, abstractmethod
from typing import Optional
from app.models.schemas import CoreState, CommandResult


class TradingService(ABC):
    @abstractmethod
    async def get_state(self) -> CoreState:
        """Получить актуальный снимок состояния ядра."""
        pass

    @abstractmethod
    async def pause_trading(self, actor: str = "Operator", reason: Optional[str] = None) -> CommandResult:
        """Приостановить торговлю (запрет новых позиций, сопровождение продолжается)."""
        pass

    @abstractmethod
    async def resume_trading(self, actor: str = "Operator") -> CommandResult:
        """Возобновить генерацию сигналов и открытие позиций."""
        pass

    @abstractmethod
    async def trigger_panic(self, actor: str = "Operator", reason: str = "Emergency flatten") -> CommandResult:
        """Аварийная ликвидация всех экспозиций и отмена всех заявок."""
        pass
