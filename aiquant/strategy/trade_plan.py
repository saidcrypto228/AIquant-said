from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TradeDecision(str, Enum):
    ACCEPTED = "ACCEPTED"
    INVALID_STOP = "INVALID_STOP"
    INVALID_TARGET = "INVALID_TARGET"
    RR_TOO_LOW = "RR_TOO_LOW"
    SIZE_TOO_SMALL = "SIZE_TOO_SMALL"


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.00035
    slippage_rate: float = 0.00010

    @property
    def round_trip_rate(self) -> float:
        return 2.0 * (self.fee_rate + self.slippage_rate)

    def estimate_round_trip_cost(self, notional: float) -> float:
        if notional < 0:
            raise ValueError("notional must be non-negative")
        return notional * self.round_trip_rate


@dataclass(frozen=True)
class RiskModel:
    risk_per_trade: float = 0.005
    max_position_notional_pct: float = 1.0
    min_position_size: float = 0.0

    def risk_budget(self, equity: float) -> float:
        if equity <= 0:
            raise ValueError("equity must be positive")
        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("risk_per_trade must be in (0, 1]")
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
        min_rr: float = 2.0,
    ) -> None:
        if min_rr <= 0:
            raise ValueError("min_rr must be positive")

        self.cost_model = cost_model or CostModel()
        self.risk_model = risk_model or RiskModel()
        self.min_rr = min_rr

    @staticmethod
    def _validate_side(side: str) -> str:
        side = side.upper()
        if side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        return side

    @staticmethod
    def _distances(
        side: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
    ) -> tuple[float, float]:
        if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
            raise ValueError("prices must be positive")

        if side == "LONG":
            risk_distance = entry - stop_loss
            reward_distance = take_profit - entry
        else:
            risk_distance = stop_loss - entry
            reward_distance = entry - take_profit

        return risk_distance, reward_distance

    def build(
        self,
        *,
        side: str,
        entry: float,
        stop_loss: float,
        take_profit: float,
        equity: float,
    ) -> TradePlan:
        side = self._validate_side(side)

        risk_distance, reward_distance = self._distances(
            side,
            entry,
            stop_loss,
            take_profit,
        )

        if risk_distance <= 0:
            return TradePlan(
                side, entry, stop_loss, take_profit,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                TradeDecision.INVALID_STOP,
                "stop_loss is on the wrong side of entry",
            )

        if reward_distance <= 0:
            return TradePlan(
                side, entry, stop_loss, take_profit,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                TradeDecision.INVALID_TARGET,
                "take_profit is on the wrong side of entry",
            )

        gross_rr = reward_distance / risk_distance
        risk_budget = self.risk_model.risk_budget(equity)

        # Position size is derived from the maximum allowed loss.
        # Stop loss risk + estimated round-trip costs must fit
        # inside the risk budget.
        cost_per_unit = entry * self.cost_model.round_trip_rate

        position_size = risk_budget / (
            risk_distance + cost_per_unit
        )

        max_notional = (
            equity * self.risk_model.max_position_notional_pct
        )

        position_size = min(
            position_size,
            max_notional / entry,
        )

        notional = position_size * entry

        estimated_costs = (
            self.cost_model.estimate_round_trip_cost(notional)
        )

        gross_risk = position_size * risk_distance
        net_risk = gross_risk + estimated_costs

        net_reward = (
            position_size * reward_distance
            - estimated_costs
        )

        net_rr = (
            net_reward / net_risk
            if net_risk > 0
            else 0.0
        )

        if net_rr < self.min_rr:
            decision = TradeDecision.RR_TOO_LOW
            reason = (
                f"net R:R {net_rr:.3f} "
                f"< minimum {self.min_rr:.3f}"
            )
        elif position_size <= self.risk_model.min_position_size:
            decision = TradeDecision.SIZE_TOO_SMALL
            reason = (
                f"position size {position_size:.8f} "
                f"<= minimum "
                f"{self.risk_model.min_position_size:.8f}"
            )
        else:
            decision = TradeDecision.ACCEPTED
            reason = None

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
            decision=decision,
            rejection_reason=reason,
        )
