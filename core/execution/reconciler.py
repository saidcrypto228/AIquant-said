"""
QVEX v10.7 — Модуль ончейн-реконсиляции и синхронизации ордеров (CRIT-01, CRIT-04).
Сверяет локальный стейт с реестром Hyperliquid L1, исключает орфанные ордера
и гарантирует наличие нативных L1-стопов на валидаторах.
"""
import asyncio
import logging
from typing import Dict, Any

logger = logging.getLogger("QVEX.Reconciliation")

class OnChainStateReconciler:
    def __init__(self, user_address: str, info_client: Any, gateway: Any):
        self.user_address = user_address
        self.info = info_client
        self.gateway = gateway

    async def perform_full_reconciliation(self) -> None:
        """Блокирующий аудит открытых позиций и нативных триггерных ордеров."""
        logger.info("[RECONCILE] Запуск ончейн-реконсиляции с распределенным реестром Hyperliquid...")

        # 1. Извлечение физических позиций и полной структуры открытых ордеров
        clearinghouse = await asyncio.to_thread(self.info.user_state, self.user_address)
        open_orders = await asyncio.to_thread(self.info.frontend_open_orders, self.user_address)

        physical_positions = {}
        for position_wrapper in clearinghouse.get("assetPositions", []):
            pos = position_wrapper["position"]
            size = float(pos["szi"])
            if abs(size) > 1e-6:
                physical_positions[pos["coin"]] = {
                    "size": size,
                    "entry_px": float(pos["entryPx"]),
                    "liquidation_px": float(pos.get("liquidationPx") or 0.0),
                    "is_long": size > 0
                }

        # 2. Индексация активных триггерных стоп-ордеров
        trigger_stops = {}
        for order in open_orders:
            if order.get("isTrigger", False):
                coin = order["coin"]
                trigger_stops[coin] = order

        # 3. Валидация защиты открытых позиций
        for coin, pos_info in physical_positions.items():
            pos_size = abs(pos_info["size"])

            if coin not in trigger_stops:
                logger.critical(
                    f"[ALERT] ОТКАЗ ЗАЩИТЫ: Позиция {coin} (объем {pos_info['size']}) не защищена стопом на L1!"
                )
                await self._restore_emergency_stop(coin, pos_info)
            else:
                trigger = trigger_stops[coin]
                trigger_size = float(trigger["sz"])
                if abs(trigger_size - pos_size) > 1e-6:
                    logger.warning(
                        f"[RECONCILE] Рассинхронизация объема {coin}: Позиция={pos_size}, Стоп={trigger_size}. Перевыставление..."
                    )
                    await asyncio.to_thread(self.gateway.exchange.cancel, coin, trigger["oid"])
                    await self.gateway.place_native_trigger_stop(
                        coin=coin,
                        is_buy=not pos_info["is_long"],
                        size=pos_size,
                        trigger_px=float(trigger["triggerPx"])
                    )

        # 4. Ликвидация орфанных стопов (ордер активен, но физическая позиция закрыта)
        for coin, trigger in trigger_stops.items():
            if coin not in physical_positions:
                logger.warning(
                    f"[RECONCILE] Обнаружен орфанный стоп-ордер по {coin} (oid={trigger['oid']}). Аннулирование..."
                )
                await asyncio.to_thread(self.gateway.exchange.cancel, coin, trigger["oid"])

        logger.info("[RECONCILE] Ончейн-реконсиляция успешно завершена.")

    async def _restore_emergency_stop(self, coin: str, pos_info: Dict[str, Any]) -> None:
        """Восстанавливает защитный биржевой стоп с аварийным буфером волатильности 1.5%."""
        mids = await asyncio.to_thread(self.info.all_mids)
        current_px = float(mids[coin])
        emergency_buffer = 0.015

        stop_px = current_px * (1.0 - emergency_buffer) if pos_info["is_long"] else current_px * (1.0 + emergency_buffer)

        await self.gateway.place_native_trigger_stop(
            coin=coin,
            is_buy=not pos_info["is_long"],
            size=abs(pos_info["size"]),
            trigger_px=round(stop_px, 4)
        )
        logger.info(f"[RECONCILE] Аварийный биржевой L1-стоп выставлен для {coin} по цене {stop_px}")
