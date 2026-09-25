from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton
)

def get_main_reply_keyboard() -> ReplyKeyboardMarkup:
    # Клавиатура без веб-кнопок: чистый терминальный пульт
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="📋 СВОДКА СИСТЕМЫ")],
            [KeyboardButton(text="💼 ПОРТФЕЛЬ И РИСК"), KeyboardButton(text="🎯 ОТКРЫТЫЕ ПОЗИЦИИ")],
            [KeyboardButton(text="🧠 МАКРО И ML-АНАЛИЗ"), KeyboardButton(text="🛑 АВАРИЙНЫЙ СБРОС")]
        ],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Команда терминала..."
    )

def get_panic_confirmation_inline() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="[ ПОДТВЕРДИТЬ СБРОС ВСЕХ СЛОТОВ ]", callback_data="confirm_panic_action")],
            [InlineKeyboardButton(text="[ ОТМЕНА ]", callback_data="cancel_panic_action")]
        ]
    )

def get_refresh_inline(action: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="[ ОБНОВИТЬ ДАННЫЕ ]", callback_data=action)]
        ]
    )
