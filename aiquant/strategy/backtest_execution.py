from __future__ import annotations

import numpy as np

from .trade_plan import TradeDecision, TradePlanEngine


def simulate_trade_plan(
    *,
    signal: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    swing_high: np.ndarray,
    swing_low: np.ndarray,
    test_idx: np.ndarray,
    capital: float = 100_000.0,
    risk_per_trade: float = 0.005,
    min_rr: float = 2.0,
    timeout_bars: int = 240,
) -> dict:
    """Execute ML signals using market-based SL/TP and risk-based sizing."""

    engine = TradePlanEngine(
        min_rr=min_rr,
    )

    equity = float(capital)
    equity_curve = np.full(len(close), equity, dtype=np.float64)

    trades = []
    wins = losses = 0
    gross_profit = gross_loss = 0.0

    test_set = set(int(x) for x in test_idx)
    i = int(test_idx[0])

    while i < len(close) - 1:
        if i not in test_set or signal[i] == 0:
            equity_curve[i] = equity
            i += 1
            continue

        side = "LONG" if signal[i] > 0 else "SHORT"

        entry_idx = i + 1
        if entry_idx >= len(close):
            break

        entry = float(open_[entry_idx])
        current_atr = float(atr[i])

        if not np.isfinite(current_atr) or current_atr <= 0:
            i += 1
            continue

        sh = float(swing_high[i]) if np.isfinite(swing_high[i]) else None
        sl = float(swing_low[i]) if np.isfinite(swing_low[i]) else None

        engine.risk_model = type(engine.risk_model)(
            risk_per_trade=risk_per_trade,
            max_position_notional_pct=engine.risk_model.max_position_notional_pct,
            min_position_size=engine.risk_model.min_position_size,
        )

        plan = engine.build_from_market(
            side=side,
            entry=entry,
            atr=current_atr,
            equity=equity,
            swing_high=sh,
            swing_low=sl,
        )

        if plan.decision != TradeDecision.ACCEPTED:
            equity_curve[i:entry_idx + 1] = equity
            i += 1
            continue

        exit_idx = min(entry_idx + timeout_bars, len(close) - 1)
        exit_price = float(close[exit_idx])
        exit_reason = "TIMEOUT"

        for j in range(entry_idx, exit_idx + 1):
            bar_high = float(high[j])
            bar_low = float(low[j])

            if side == "LONG":
                hit_sl = bar_low <= plan.stop_loss
                hit_tp = bar_high >= plan.take_profit
            else:
                hit_sl = bar_high >= plan.stop_loss
                hit_tp = bar_low <= plan.take_profit

            # Conservative rule: if both are touched in one 1m bar,
            # assume SL was hit first.
            if hit_sl:
                exit_idx = j
                exit_price = plan.stop_loss
                exit_reason = "SL"
                break

            if hit_tp:
                exit_idx = j
                exit_price = plan.take_profit
                exit_reason = "TP"
                break

        if side == "LONG":
            gross_pnl = (
                exit_price - plan.entry
            ) * plan.position_size
        else:
            gross_pnl = (
                plan.entry - exit_price
            ) * plan.position_size

        costs = plan.estimated_costs
        pnl = gross_pnl - costs

        equity = max(equity + pnl, 1.0)

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += abs(pnl)

        trades.append(
            {
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "side": side,
                "entry": plan.entry,
                "stop_loss": plan.stop_loss,
                "take_profit": plan.take_profit,
                "exit": exit_price,
                "position_size": plan.position_size,
                "notional": plan.notional,
                "gross_rr": plan.gross_rr,
                "net_rr": plan.net_rr,
                "pnl": pnl,
                "exit_reason": exit_reason,
            }
        )

        equity_curve[i:exit_idx + 1] = equity
        i = exit_idx + 1

    equity_curve[-1] = equity

    total_trades = len(trades)
    win_rate = (
        wins / total_trades
        if total_trades
        else 0.0
    )

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else float("inf") if gross_profit > 0 else 0.0
    )

    return {
        "final_equity": equity,
        "return_pct": (equity / capital - 1.0) * 100.0,
        "equity_curve": equity_curve,
        "trades": trades,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
    }