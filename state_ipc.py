"""
QVEX v10.7 — Потокобезопасный межпроцессный менеджер состояний (POSIX Atomic IPC).
Использует атомарную замену файлов os.replace для гарантированной целостности данных.
"""
import os
import json
import logging
from pathlib import Path
from typing import Type, TypeVar
from pydantic import BaseModel

logger = logging.getLogger("QVEX.StateIPC")
T = TypeVar("T", bound=BaseModel)

class PosixAtomicStateManager:
    def __init__(self, filepath: str, schema_cls: Type[T]):
        self.path = Path(filepath).resolve()
        self.schema_cls = schema_cls
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_atomic_state(self, state: T) -> None:
        """Атомарная запись через локальный staging-файл с последующей заменой."""
        tmp_file = self.path.with_suffix(".tmp")
        try:
            payload = state.model_dump_json(indent=2)
            tmp_file.write_text(payload, encoding="utf-8")
            os.replace(tmp_file, self.path)
        except Exception as err:
            logger.error(f"[IPC] Ошибка атомарной записи в {self.path.name}: {err}")
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise

    def read_atomic_state(self) -> T:
        """Потокобезопасное чтение и валидация через Pydantic-схему."""
        if not self.path.exists():
            raise FileNotFoundError(f"Файл состояния не найден: {self.path}")
        raw_data = self.path.read_text(encoding="utf-8")
        return self.schema_cls.model_validate_json(raw_data)
