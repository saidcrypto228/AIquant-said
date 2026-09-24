from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from enum import Enum


class TradeDecision(str, Enum):
    ACCEPTED = "ACCEPTED"
    STOP_TOO_TIGHT = "STOP_TOO_TIGHT"
    INVALID_STOP = "INVALID_STOP"
    INVALID_TARGET = "INVALID_TARGET"
    RR_TOO_LOW = "RR_TOO_LOW"
    SIZE_TOO_SMALL = "SIZE_TOO_SMALL"


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.00035           # Taker fee 0.035%
    slippage_entry: float = 0.00010     # 0.010%
    slippage_stop: float = 0.00015      # 0.015%

    @property
    def round_trip_rate(self) -> float:
        return (2.0 * self.fee_rate) + self.slippage_entry + self.slippage_stop  # 0.095%

    def estimate_round_trip_cost(self, notional: float) -> float:
        return notional * self.round_trip_rate


@dataclass(frozen=True)
class RiskModel:
    risk_per_trade: float = 0.005       # 0.5% ($500 при $100k)
    max_fee_drag_ratio: float = 0.10    # Комиссия не более 10% от риска сделки
    max_leverage: float = 1.0           # Жесткий потолок плеча 1.0x (ликвидация оверлевериджа)

    def risk_budget(self, equity: float) -> float:
        return equity * self.risk_per_trade


@dataclass(frozen=True)
class TradePlan:
    side: str
    entry: float
    stop_loss: float
    take_profit: float
    gross_rr: float
    net_rr: float
    estimated_costs: float
    risk_budget: float
    position_size: float
    notional: float
    decision: TradeDecision
    rejection_reason: str | None = None


class TradePlanEngine:
    def __init__(
        self,
        cost_model: CostModel | None = None,
        risk_model: RiskModel | None = None,
        min_rr: float = 0.5,
        target_rr: float = 1.40,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.risk_model = risk_model or RiskModel()
        self.min_rr = min_rr
        self.target_rr = target_rr
        # Инвариант: стоп обязан быть шире комиссии минимум в 10 раз (0.095% / 0.10 = 0.95%)
        self.min_safe_stop_pct = max(0.0090, self.cost_model.round_trip_rate / self.risk_model.max_fee_drag_ratio)

    def build_from_market(
        self,
        *,
        side: str,
        entry: float,
        atr: float,
        equity: float,
        swing_high: float | None = None,
        swing_low: float | None = None,
        atr_stop_mult: float = 2.0,
        **kwargs,
    ) -> TradePlan:
        side = side.upper()

        # Волатильностный стоп с жестким физическим порогом 0.90%
        calculated_stop_dist = atr * atr_stop_mult
        min_stop_dist = entry * self.min_safe_stop_pct
        stop_dist = max(calculated_stop_dist, min_stop_dist)

        target_dist = stop_dist * self.target_rr

        if side == "LONG":
            stop_loss = entry - stop_dist
            take_profit = entry + target_dist
        else:
            stop_loss = entry + stop_dist
            take_profit = entry - target_dist

        return self.build(
            side=side,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            equity=equity,
        )

    def build(
        self,
        *,
        side: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        equity: float,
    ) -> TradePlan:
        side = side.upper()
        risk_dist = (entry - stop_loss) if side == "LONG" else (stop_loss - entry)
        reward_dist = (take_profit - entry) if side == "LONG" else (entry - take_profit)

        if risk_dist <= 0:
            return TradePlan(side, entry, stop_loss, take_profit, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             TradeDecision.INVALID_STOP, "Stop loss wrong side of entry")

        stop_dist_pct = risk_dist / entry
        if stop_dist_pct < self.min_safe_stop_pct:
            return TradePlan(side, entry, stop_loss, take_profit, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             TradeDecision.STOP_TOO_TIGHT,
                             f"Stop {stop_dist_pct:.4%} < SafeFloor {self.min_safe_stop_pct:.2%}")

        gross_rr = reward_dist / risk_dist
        if gross_rr < self.min_rr:
            return TradePlan(side, entry, stop_loss, take_profit, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                             TradeDecision.RR_TOO_LOW, f"RR {gross_rr:.2f} < min_rr {self.min_rr:.2f}")

        risk_budget = self.risk_model.risk_budget(equity)
        friction_rate = self.cost_model.round_trip_rate

        # Регуляризованный сайзинг: комиссия включена в знаменатель
        effective_loss_per_unit = risk_dist + (entry * friction_rate)
        q_risk = risk_budget / effective_loss_per_unit

        # Ограничение максимального плеча (1.0x от капитала)
        q_max_lev = (equity * self.risk_model.max_leverage) / entry
        position_size = min(q_risk, q_max_lev)
        notional = position_size * entry

        estimated_costs = notional * friction_rate
        net_rr = (reward_dist * position_size - estimated_costs) / risk_budget

        return TradePlan(
            side=side,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            gross_rr=gross_rr,
            net_rr=net_rr,
            estimated_costs=estimated_costs,
            risk_budget=risk_budget,
            position_size=position_size,
            notional=notional,
            decision=TradeDecision.ACCEPTED,
            rejection_reason=None,
        )


# =====================================================================
# CONVICTION RISK ENGINE: Динамический сайзинг по уверенности ML
# =====================================================================
class ConvictionRiskEngine:
    def __init__(
        self,
        base_risk_pct: float = 0.010,       # Базовый риск 1.0% ($100 на $10k счета)
        min_risk_pct: float = 0.005,        # Пол риска 0.5%
        max_risk_pct: float = 0.020,        # Потолок 2.0% (защита от переобучения)
        max_account_leverage: float = 1.5,  # Предельное плечо к балансу
        friction_pct: float = 0.00095,      # 0.095% taker roundtrip + spread
        gamma: float = 1.5,                 # Степень выпуклости
        min_stop_distance_pct: float = 0.0055
    ):
        self.r_base = base_risk_pct
        self.r_min = min_risk_pct
        self.r_max = max_risk_pct
        self.max_lev = max_account_leverage
        self.friction = friction_pct
        self.gamma = gamma
        self.min_stop = min_stop_distance_pct

    def compute_conviction_risk(
        self,
        calibrated_prob: float,
        stop_dist_pct: float,
        target_dist_pct: float
    ) -> float:
        denom = target_dist_pct + stop_dist_pct
        if denom <= 0:
            return 0.0
        p_be = (stop_dist_pct + self.friction) / denom
        safety_threshold = p_be + 0.04  # 4% буфер над безубытком

        if calibrated_prob < safety_threshold:
            return 0.0

        conviction_span = max(0.0, (calibrated_prob - safety_threshold) / (1.0 - safety_threshold))
        scaled_risk = self.r_base + (self.r_max - self.r_base) * (conviction_span ** self.gamma)
        return float(np.clip(scaled_risk, self.r_min, self.r_max))

    def size_order(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        calibrated_prob: float,
        target_rr: float = 1.25
    ) -> dict:
        stop_dist_abs = abs(entry_price - stop_price)
        stop_dist_pct = stop_dist_abs / entry_price
        target_dist_pct = stop_dist_pct * target_rr

        if stop_dist_pct < self.min_stop:
            return {"approved": False, "reason": "STOP_BELOW_NOISE_FLOOR"}

        risk_pct = self.compute_conviction_risk(calibrated_prob, stop_dist_pct, target_dist_pct)
        if risk_pct <= 0:
            return {"approved": False, "reason": "INSUFFICIENT_PROBABILITY_EDGE"}

        risk_dollars = equity * risk_pct
        effective_loss_per_unit = stop_dist_pct + self.friction
        target_notional = risk_dollars / effective_loss_per_unit
        max_notional_cap = equity * self.max_lev

        final_notional = min(target_notional, max_notional_cap)
        quantity = final_notional / entry_price

        if final_notional < 11.0:
            return {"approved": False, "reason": "NOTIONAL_BELOW_EXCHANGE_MIN"}

        return {
            "approved": True,
            "quantity": quantity,
            "notional": final_notional,
            "allocated_risk_pct": risk_pct,
            "effective_leverage": final_notional / equity
        }