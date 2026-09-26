# ТЕХНИЧЕСКАЯ СПЕЦИФИКАЦИЯ TELEGRAM-БОТА QVEX v10.7

## Назначение
Изолированный ChatOps-клиент для мониторинга и управления ядром Hyperliquid L1.
Бот НЕ содержит торговой логики и взаимодействует с ядром ИСКЛЮЧИТЕЛЬНО через IPC-файлы в `data/`.

## Архитектура IPC
1. ЧТЕНИЕ ТЕЛЕМЕТРИИ (data/bot_state.json):
   - Использовать `PosixAtomicStateManager` из `state_ipc.py`.
   - Модель данных `StateSnapshot` из `state_schema.py`.
   - Поля: total_equity, free_margin, btc_price, market_regime, positions.
2. ЗАПИСЬ КОМАНД (data/trading_control.json):
   - Использовать `ControlStateManager` из `control_ipc.py`.
   - Методы: pause_trading(), resume_trading(), trigger_panic().

## Функционал интерфейса
- Reply-кнопки:
  • 📊 Сводка системы (/status)
  • 💼 Портфель и риск (/risk)
  • 🎯 Открытые позиции (/positions)
  • ⚙️ Управление (/control)
- Inline-кнопки:
  • Пауза / Возобновление работы
  • Обновление данных в сообщении
  • Двухэтапная экстренная ликвидация (EMERGENCY PANIC)
- Безопасность:
  • Строгий Whitelist по `TELEGRAM_ADMIN_ID` из `.env`.
