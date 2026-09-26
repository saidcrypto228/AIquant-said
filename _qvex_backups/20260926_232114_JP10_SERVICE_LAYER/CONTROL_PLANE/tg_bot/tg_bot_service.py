import sys
from pathlib import Path

# Гарантия корректных импортов из корня и внутри пакета tg_bot
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))
if str(_root_dir) not in sys.path:
    sys.path.insert(0, str(_root_dir))

import asyncio
import json
import logging
import aiohttp
from aiohttp.resolver import ThreadedResolver
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import MenuButtonWebApp, WebAppInfo

from tg_config import settings
from tg_handlers import router
from tg_auth import get_whitelist

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s: %(message)s")
logger = logging.getLogger("TgService")

class WindowsSafeSession(AiohttpSession):
    async def create_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(resolver=ThreadedResolver())
            self._session = aiohttp.ClientSession(
                connector=connector,
                json_serialize=self.json_dumps
            )
        return self._session

async def send_broadcast(bot: Bot, html_text: str):
    recipients = set(get_whitelist())
    if settings.ADMIN_ID:
        recipients.add(settings.ADMIN_ID)
    for uid in recipients:
        try:
            await bot.send_message(chat_id=uid, text=html_text)
        except Exception:
            pass

async def monitor_bot_state(bot: Bot):
    logger.info("Воркер мониторинга bot_state.json запущен.")
    last_positions = None
    last_regime = None

    while True:
        try:
            await asyncio.sleep(2.0)
            if not settings.state_path.exists():
                continue
            try:
                state = json.loads(settings.state_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            positions = {p["symbol"]: p for p in state.get("positions", [])}
            regime = state.get("market_regime", {})

            if last_positions is None or last_regime is None:
                last_positions = positions
                last_regime = regime
                continue

            # Новая позиция
            for sym, pos in positions.items():
                if sym not in last_positions:
                    msg = (
                        "**[ORDER EXECUTION] НОВАЯ ПОЗИЦИЯ ОТКРЫТА**\n"
                        "`────────────────────────────────────`\n"
                        f"• ТИКЕР     : **{pos.get('symbol')}-PERP** ({pos.get('side')})\n"
                        f"• ВХОД      : `${pos.get('entry_price', 0.0):,.4f}`\n"
                        f"• СТОП-ЛОСС : `${pos.get('sl_price', 0.0):,.4f}`\n"
                        f"• РИСК      : `1.00% ЭКВИТИ`"
                    )
                    await send_broadcast(bot, msg)
                else:
                    prev_pos = last_positions[sym]
                    if not prev_pos.get("is_breakeven") and pos.get("is_breakeven"):
                        msg = (
                            "**[RISK UPDATE] СТОП-ЛОСС ПЕРЕВЕДЕН В БЕЗУБЫТОК**\n"
                            "`────────────────────────────────────`\n"
                            f"• ТИКЕР     : **{pos.get('symbol')}-PERP**\n"
                            f"• НОВЫЙ SL  : `${pos.get('sl_price', 0.0):,.4f}`\n"
                            "• РИСК      : **0.00 USD [ZERO RISK ACTIVE]**"
                        )
                        await send_broadcast(bot, msg)

            # Закрытие сделки
            for sym, old_pos in last_positions.items():
                if sym not in positions:
                    pnl_usd = old_pos.get("pnl_usd", 0.0)
                    pnl_pct = old_pos.get("pnl_pct", 0.0)
                    sign = "+" if pnl_usd >= 0 else ""
                    res_tag = "[PROFIT]" if pnl_usd >= 0 else "[LOSS]"
                    msg = (
                        f"**[POSITION CLOSED] {res_tag}**\n"
                        "`────────────────────────────────────`\n"
                        f"• ТИКЕР     : **{old_pos.get('symbol')}-PERP**\n"
                        f"• РЕЗУЛЬТАТ : **{sign}${pnl_usd:,.2f} ({sign}{pnl_pct:.2f}%)**"
                    )
                    await send_broadcast(bot, msg)

            last_positions = positions
            last_regime = regime

        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(5.0)

async def main():
    token = settings.token
    if not token or len(token) < 20:
        logger.error("BOT_TOKEN не задан в .env!")
        return

    session = WindowsSafeSession()
    bot = Bot(token=token, session=session, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher()
    dp.include_router(router)

    # Регистрация нативной кнопки Mini App в нижнем левом углу
    if settings.TMA_URL.startswith("https://"):
        try:
            await bot.set_chat_menu_button(
                menu_button=MenuButtonWebApp(
                    text="📊 Терминал",
                    web_app=WebAppInfo(url=settings.TMA_URL)
                )
            )
            logger.info("Нативная кнопка WebApp Menu зарегистрирована.")
        except Exception:
            pass

    monitor_task = asyncio.create_task(monitor_bot_state(bot))
    try:
        logger.info("Запуск aiogram Dispatcher Polling...")
        await dp.start_polling(bot)
    finally:
        monitor_task.cancel()
        await bot.session.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
