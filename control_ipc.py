"""
QVEX v10.7 — Менеджер удаленного управления торговлей (ChatOps Control Bus).
Обеспечивает атомарную передачу команд между Telegram-ботом и торговым ядром.
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from pydantic import BaseModel, Field

logger = logging.getLogger("QVEX.ControlIPC")

class TradingControlState(BaseModel):
    trading_enabled: bool = Field(default=True, description="Разрешение на открытие новых позиций")
    panic_requested: bool = Field(default=False, description="Флаг экстренной ликвидации портфеля")
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_command_by: str = Field(default="system")
    message: str = Field(default="Штатный режим")

class ControlStateManager:
    def __init__(self, filepath: str = "data/trading_control.json"):
        self.path = Path(filepath).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.set_state(TradingControlState())

    def get_state(self) -> TradingControlState:
        try:
            if not self.path.exists():
                return TradingControlState()
            raw = self.path.read_text(encoding="utf-8")
            return TradingControlState.model_validate_json(raw)
        except Exception as e:
            logger.error(f"Ошибка чтения флагов управления: {e}")
            return TradingControlState()

    def set_state(self, state: TradingControlState) -> None:
        state.updated_at = datetime.now(timezone.utc).isoformat()
        temp_path = self.path.with_suffix(".tmp")
        temp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        temp_path.replace(self.path)

    def pause_trading(self, admin_tag: str = "Telegram Admin") -> TradingControlState:
        state = self.get_state()
        state.trading_enabled = False
        state.last_command_by = admin_tag
        state.message = "Торговля приостановлена пользователем (новые сделки заблокированы)"
        self.set_state(state)
        return state

    def resume_trading(self, admin_tag: str = "Telegram Admin") -> TradingControlState:
        state = self.get_state()
        state.trading_enabled = True
        state.last_command_by = admin_tag
        state.message = "Торговля активна (генерация сигналов включена)"
        self.set_state(state)
        return state

    def trigger_panic(self, admin_tag: str = "Telegram Admin") -> TradingControlState:
        state = self.get_state()
        state.panic_requested = True
        state.trading_enabled = False
        state.last_command_by = admin_tag
        state.message = "АКТИВИРОВАН РЕЖИМ PANIC: экстренный сброс всех позиций!"
        self.set_state(state)
        return state

    def clear_panic(self) -> None:
        state = self.get_state()
        state.panic_requested = False
        self.set_state(state)
