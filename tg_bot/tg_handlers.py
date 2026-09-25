import json
import time
from datetime import datetime, timezone
from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery
from aiogram.exceptions import TelegramBadRequest

from tg_config import settings
from tg_auth import is_user_allowed
from tg_keyboards import (
    get_main_reply_keyboard,
    get_panic_confirmation_inline,
    get_refresh_inline
)

router = Router(name="institutional_router")

async def guard_access(message: Message) -> bool:
    user_id = message.from_user.id if message.from_user else 0
    if is_user_allowed(user_id):
        return True
    await message.answer(
        "`[ДОСТУП ЗАПРЕЩЕН]`\n"
        f"Идентификатор пользователя: `{user_id}`\n"
        "Устройство не авторизовано в whitelist.json."
    )
    return False

def get_state() -> dict:
    if not settings.state_path.exists():
        return {}
    try:
        return json.loads(settings.state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def render_system_summary() -> str:
    s = get_state()
    acc = s.get("account", {})
    reg = s.get("market_regime", {})
    pos = s.get("positions", [])
    if isinstance(pos, dict):
        pos = list(pos.values())

    now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    eq = acc.get("equity", 0.0)
    bal = acc.get("total_balance", 0.0)
    free = acc.get("free_margin", 0.0)
    status_reg = reg.get("status", "BULL")
    slope = reg.get("slope", 0.0)
    storm = "АКТИВЕН [БЛОКИРОВКА ВХОДОВ]" if reg.get("storm_filter_active") else "ВЫКЛ [НОРМА]"

    # ML Verdict
    ml_verdict = "LONG BIAS (ВХОДЫ РАЗРЕШЕНЫ)" if status_reg == "BULL" else "DEFENSIVE (СОКРАЩЕНИЕ РИСКА)"

    lines = [
        "**QVEX TERMINAL | СИСТЕМНЫЙ ДАЙДЖЕСТ**",
        "`────────────────────────────────────`",
        f"КОНТУР      : **HYPERLIQUID DEX**",
        f"ВРЕМЯ СРЕЗА : `{now_utc}`",
        "",
        "**[ ДЕПОЗИТ И СЛОТЫ ]**",
        f"ЭКВИТИ      : `${eq:,.2f}`",
        f"БАЛАНС      : `${bal:,.2f}`",
        f"СВ. МАРЖА   : `${free:,.2f}`",
        f"СЛОТЫ       : **{len(pos)} / 2 АКТИВНО**",
        "",
        "**[ МАКРО-СТАТУС И ML-МОДЕЛЬ ]**",
        f"РЕЖИМ BTC   : **{status_reg}** (SLOPE: `{slope:+.2f}`)",
        f"ШТОРМ-ФИЛЬТР: `{storm}`",
        f"ВЕРДИКТ ML  : `{ml_verdict}`",
        "",
        "**[ АКТИВНЫЕ ПОЗИЦИИ ]**"
    ]

    if not pos:
        lines.append("*Активных сделок нет. Слоты в режиме ожидания триггеров.*")
    else:
        for idx, p in enumerate(pos, start=1):
            pnl_usd = p.get("pnl_usd", 0.0)
            pnl_pct = p.get("pnl_pct", 0.0)
            sign = "+" if pnl_usd >= 0 else ""
            be = " [BE PROTECTED]" if p.get("is_breakeven") else ""
            lines.append(
                f"**[{idx}] {p.get('symbol')}-PERP | {p.get('side')}**{be}\n"
                f"    Вход: `${p.get('entry_price', 0.0):,.4f}` | Тек: `${p.get('current_price', 0.0):,.4f}`\n"
                f"    PnL : **{sign}${pnl_usd:,.2f} ({sign}{pnl_pct:.2f}%)** | SL: `${p.get('sl_price', 0.0):,.4f}`"
            )
    lines.append("`────────────────────────────────────`")
    return "\n".join(lines)

def render_portfolio_risk() -> str:
    s = get_state()
    acc = s.get("account", {})
    eq = acc.get("equity", 0.0)
    bal = acc.get("total_balance", 0.0)
    free = acc.get("free_margin", 0.0)
    used = max(0.0, eq - free)
    utilization = (used / eq * 100.0) if eq > 0 else 0.0

    return (
        "**QVEX | АУДИТ ДЕПОЗИТА И МАТРИЦА РИСКА**\n"
        "`────────────────────────────────────`\n"
        f"ЭКВИТИ ПОРТФЕЛЯ : `${eq:,.2f} USD`\n"
        f"БАЛАНС СЧЕТА    : `${bal:,.2f} USD`\n"
        f"СВОБОДНАЯ МАРЖА : `${free:,.2f} USD`\n"
        f"ЗАДЕЙСТВОВАНО   : `${used:,.2f} USD ({utilization:.1f}%)`\n"
        "`────────────────────────────────────`\n"
        "**[ ПАРАМЕТРЫ РИСК-МЕНЕДЖМЕНТА ]**\n"
        "• ЛИМИТ СЛОТОВ   : **2 ПАРАЛЛЕЛЬНЫХ (HARD-CODED)**\n"
        "• РИСК НА СДЕЛКУ : **1.00% ОТ СУММАРНОГО ЭКВИТИ**\n"
        "• ТРЕЙЛИНГ-СТОП  : **АТР-ДИНАМИЧЕСКИЙ**\n"
        "• БЕЗУБЫТОК (BE) : **АВТОМАТИЧЕСКИ ПРИ ДОСТИЖЕНИИ +1.5R**\n"
        "`────────────────────────────────────`"
    )

def render_positions() -> str:
    s = get_state()
    pos = s.get("positions", [])
    if isinstance(pos, dict):
        pos = list(pos.values())

    lines = [
        "**QVEX | СЛОТЫ ПОРТФЕЛЯ (ИНСТИТУЦИОНАЛЬНЫЙ СРЕЗ)**",
        "`────────────────────────────────────`"
    ]

    for i in range(2):
        slot_num = i + 1
        if i < len(pos):
            p = pos[i]
            pnl_usd = p.get("pnl_usd", 0.0)
            pnl_pct = p.get("pnl_pct", 0.0)
            sign = "+" if pnl_usd >= 0 else ""
            status_be = "ЗАЩИЩЕН (В БЕЗУБЫТКЕ)" if p.get("is_breakeven") else "СТОП АКТИВЕН"
            lines.extend([
                f"**[СЛОТ {slot_num}] {p.get('symbol')}-PERP | {p.get('side')}**",
                f"├ ТОЧКА ВХОДА : `${p.get('entry_price', 0.0):,.4f}`",
                f"├ РЫНОЧНАЯ    : `${p.get('current_price', 0.0):,.4f}`",
                f"├ СТОП-ЛОСС   : `${p.get('sl_price', 0.0):,.4f}` [{status_be}]",
                f"└ PnL ПОЗИЦИИ : **{sign}${pnl_usd:,.2f} ({sign}{pnl_pct:.2f}%)**",
                ""
            ])
        else:
            lines.extend([
                f"**[СЛОТ {slot_num}] СВОБОДЕН**",
                "└ СТАТУС      : Мониторинг пула Meta-Model (Ожидание сигнала)",
                ""
            ])

    lines.append("`────────────────────────────────────`")
    return "\n".join(lines)

def render_macro_ml() -> str:
    s = get_state()
    reg = s.get("market_regime", {})
    trig = s.get("active_triggers", [])

    btc_p = reg.get("btc_price", 0.0)
    status = reg.get("status", "BULL")
    slope = reg.get("slope", 0.0)
    storm = "АКТИВЕН (ВОЛАТИЛЬНОСТЬ ПРЕВЫШЕНА)" if reg.get("storm_filter_active") else "ВЫКЛ [НОРМА]"

    bias_desc = "БЫЧИЙ ТРЕНД (ПРИОРИТЕТ ЛОНГ-СЕТАПАМ)" if status == "BULL" else "МЕДВЕЖИЙ ТРЕНД (КОНСЕРВАТИВНЫЙ РЕЖИМ)"

    lines = [
        "**QVEX | МАКРО-АНАЛИЗ И ПРОГНОЗ МОДЕЛИ ML**",
        "`────────────────────────────────────`",
        f"БАЗОВЫЙ АКТИВ : **BTC/USD (${btc_p:,.1f})**",
        f"РЕЖИМ РЫНКА   : **{status} REGIME**",
        f"НАКЛОН (SLOPE): `{slope:+.4f}`",
        f"ШТОРМ-ФИЛЬТР  : `{storm}`",
        "",
        "**[ ОЦЕНКА ИСКУССТВЕННОГО ИНТЕЛЛЕКТА (ML) ]**",
        f"АРХИТЕКТУРА   : **Linear Meta-Model v10.7 Hardened**",
        f"НАПРАВЛЕНИЕ   : `{bias_desc}`",
        f"ДОПУСК СДЕЛОК : **РАЗРЕШЕН В ОБА СЛОТА**",
        "`────────────────────────────────────`",
        "**ПРЕ-ЭНТРИ РАДАР (КАНДИДАТЫ НА СЛЕДУЮЩИЙ ТАКТ):**"
    ]

    if not trig:
        lines.append("*Кандидаты в пределах расчетной дельты не зафиксированы.*")
    else:
        for t in trig:
            lines.append(
                f"• **{t.get('symbol')}-PERP** ({t.get('side')}) : "
                f"Триггер `${t.get('trigger_price', 0.0):,.2f}` | Дельта: `{t.get('distance_pct', 0.0):.2f}%`"
            )
    lines.append("`────────────────────────────────────`")
    return "\n".join(lines)

# ХЭНДЛЕРЫ
@router.message(CommandStart())
async def cmd_start(msg: Message):
    if not await guard_access(msg): return
    welcome = (
        "**QVEX TRADING SYSTEM v10.7 (INSTITUTIONAL BASELINE)**\n"
        "`────────────────────────────────────`\n"
        "Терминальный доступ к торговому комплексу Hyperliquid.\n"
        "Управление графиками доступно через нативную кнопку **Терминал** слева внизу.\n\n"
        "Используйте клавиатуру для запроса телеметрии ядра."
    )
    await msg.answer(welcome, reply_markup=get_main_reply_keyboard())

@router.message(F.text == "📋 СВОДКА СИСТЕМЫ")
async def msg_summary(msg: Message):
    if not await guard_access(msg): return
    await msg.answer(render_system_summary(), reply_markup=get_refresh_inline("refresh_summary"))

@router.message(F.text == "💼 ПОРТФЕЛЬ И РИСК")
async def msg_risk(msg: Message):
    if not await guard_access(msg): return
    await msg.answer(render_portfolio_risk(), reply_markup=get_refresh_inline("refresh_risk"))

@router.message(F.text == "🎯 ОТКРЫТЫЕ ПОЗИЦИИ")
async def msg_positions(msg: Message):
    if not await guard_access(msg): return
    await msg.answer(render_positions(), reply_markup=get_refresh_inline("refresh_positions"))

@router.message(F.text == "🧠 МАКРО И ML-АНАЛИЗ")
async def msg_macro(msg: Message):
    if not await guard_access(msg): return
    await msg.answer(render_macro_ml(), reply_markup=get_refresh_inline("refresh_macro"))

@router.message(F.text == "🛑 АВАРИЙНЫЙ СБРОС")
async def msg_panic_prompt(msg: Message):
    if not await guard_access(msg): return
    warn = (
        "**[!] ИНИЦИАЦИЯ ПРОТОКОЛА АВАРИЙНОЙ ОСТАНОВКИ**\n"
        "`────────────────────────────────────`\n"
        "ВНИМАНИЕ: Все открытые позиции на Hyperliquid будут\n"
        "немедленно закрыты рыночными ордерами (MARKET SLIPPAGE).\n\n"
        "Подтвердить отправку сигнала экстренного сброса?"
    )
    await msg.answer(warn, reply_markup=get_panic_confirmation_inline())

# CALLBACKS
@router.callback_query(F.data == "refresh_summary")
async def cb_sum(q: CallbackQuery):
    try:
        await q.message.edit_text(render_system_summary(), reply_markup=get_refresh_inline("refresh_summary"))
        await q.answer("Дайджест обновлен")
    except TelegramBadRequest:
        await q.answer("Изменений нет")

@router.callback_query(F.data == "refresh_risk")
async def cb_risk(q: CallbackQuery):
    try:
        await q.message.edit_text(render_portfolio_risk(), reply_markup=get_refresh_inline("refresh_risk"))
        await q.answer("Риски пересчитаны")
    except TelegramBadRequest:
        await q.answer("Изменений нет")

@router.callback_query(F.data == "refresh_positions")
async def cb_pos(q: CallbackQuery):
    try:
        await q.message.edit_text(render_positions(), reply_markup=get_refresh_inline("refresh_positions"))
        await q.answer("Позиции проверены")
    except TelegramBadRequest:
        await q.answer("Изменений нет")

@router.callback_query(F.data == "refresh_macro")
async def cb_mac(q: CallbackQuery):
    try:
        await q.message.edit_text(render_macro_ml(), reply_markup=get_refresh_inline("refresh_macro"))
        await q.answer("Макро-данные обновлены")
    except TelegramBadRequest:
        await q.answer("Изменений нет")

@router.callback_query(F.data == "confirm_panic_action")
async def cb_panic_confirm(q: CallbackQuery):
    now = time.time()
    payload = {"triggered_at": now, "source": f"tg_admin_{q.from_user.id}"}
    settings.panic_trigger_path.write_text(json.dumps(payload), encoding="utf-8")

    # Мгновенная очистка позиций в bot_state
    if settings.state_path.exists():
        try:
            st = json.loads(settings.state_path.read_text(encoding="utf-8"))
            st["positions"] = []
            st["timestamp"] = now
            settings.state_path.write_text(json.dumps(st, indent=2), encoding="utf-8")
        except Exception:
            pass

    await q.message.edit_text(
        "**[СИГНАЛ ПЕРЕДАН] АВАРИЙНЫЙ KILL-SWITCH АКТИВИРОВАН.**\n"
        "Торговое ядро закрывает позиции рыночными ордерами."
    )
    await q.answer("СБРОС ВЫПОЛНЕН", show_alert=True)

@router.callback_query(F.data == "cancel_panic_action")
async def cb_panic_cancel(q: CallbackQuery):
    await q.message.edit_text("`[ОТМЕНА] Сигнал аварийной остановки отклонен.`")
    await q.answer("Отменено")
