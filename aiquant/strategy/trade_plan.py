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
    stop_slippage_rate: float = 0.00015

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
    max_leverage: float = 2.5
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
        min_rr: float = 0.5,
    ) -> None:
        self.cost_model = cost_model or CostModel()
        self.risk_model = risk_model or RiskModel()
        self.min_rr = min_rr

    def build_from_market(
        self,
        *,
        side: str,
        entry: float,
        atr: float,
        equity: float,
        swing_high: float | None = None,
        swing_low: float | None = None,
        atr_stop_mult: float = 1.8,
        atr_target_mult: float = 2.25,
    ) -> TradePlan:
        side = side.upper()
        # Полная синхронизация с compute_triple_barrier_labels (stop=1.8*ATR, min_stop=0.30%, target_ratio=1.25)
        base_stop_dist = max(entry * 0.0030, atr * atr_stop_mult)
        target_ratio = 1.25

        # Strictly aligned with compute_triple_barrier_labels
        if side == "LONG":
            stop_loss = entry - base_stop_dist
            take_profit = entry + base_stop_dist * target_ratio
        else:
            stop_loss = entry + base_stop_dist
            take_profit = entry - base_stop_dist * target_ratio

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
            return TradePlan(side, entry, stop_loss, take_profit, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, TradeDecision.INVALID_STOP, "stop_loss wrong side")
        if reward_dist <= 0:
            return TradePlan(side, entry, stop_loss, take_profit, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, TradeDecision.INVALID_TARGET, "take_profit wrong side")

        gross_rr = reward_dist / risk_dist
        risk_budget = self.risk_model.risk_budget(equity)
        cost_per_unit = entry * (self.cost_model.round_trip_rate + self.cost_model.stop_slippage_rate)

        q_risk = risk_budget / (risk_dist + cost_per_unit)
        q_max_lev = (equity * self.risk_model.max_leverage) / entry
        q_max_notional = (equity * self.risk_model.max_position_notional_pct) / entry
        position_size = min(q_risk, q_max_lev, q_max_notional)
        notional = position_size * entry

        estimated_costs = self.cost_model.estimate_round_trip_cost(notional)
        net_risk = (position_size * risk_dist) + estimated_costs
        net_reward = (position_size * reward_dist) - estimated_costs
        net_rr = net_reward / net_risk if net_risk > 0 else 0.0

        if net_rr < self.min_rr:
            decision = TradeDecision.RR_TOO_LOW
            reason = f"net R:R {net_rr:.3f} < min {self.min_rr:.3f}"
        elif position_size <= self.risk_model.min_position_size:
            decision = TradeDecision.SIZE_TOO_SMALL
            reason = f"size {position_size:.6f} <= min {self.risk_model.min_position_size:.6f}"
        else:
            decision = TradeDecision.ACCEPTED
            reason = None

        return TradePlan(
            side, entry, stop_loss, take_profit, gross_rr, net_rr,
            estimated_costs, risk_budget, position_size, notional, decision, reason
        )
