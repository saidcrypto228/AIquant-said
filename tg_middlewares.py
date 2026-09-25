from __future__ import annotations
import logging
from typing import Any, Awaitable, Callable
from aiogram import BaseMiddleware
from aiogram.types import CallbackQuery, Message, TelegramObject

logger = logging.getLogger(__name__)
Handler = Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]]

class AdminOnlyMiddleware(BaseMiddleware):
    def __init__(self, admin_ids: set[int]) -> None:
        self._admin_ids = frozenset(admin_ids)

    async def __call__(
        self,
        handler: Handler,
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = getattr(event, "from_user", None)
        if user is None:
            return await handler(event, data)

        # Если список пуст или содержит 0 — разрешаем доступ и логируем ID для настройки
        if not self._admin_ids or 0 in self._admin_ids:
            logger.info("Admin whitelist не ограничен. Авторизован user_id=%s (@%s)", user.id, user.username)
            return await handler(event, data)

        if user.id in self._admin_ids:
            return await handler(event, data)

        logger.warning("Попытка доступа: user_id=%s username=%r event=%s", user.id, user.username, type(event).__name__)
        warn_text = f"⛔ Доступ запрещен. Ваш ID: {user.id} не в белом списке."
        if isinstance(event, Message):
            await event.answer(warn_text)
        elif isinstance(event, CallbackQuery):
            await event.answer(warn_text, show_alert=True)
        return None
