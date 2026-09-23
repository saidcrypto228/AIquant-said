from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
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
    min_rr: float = 0.5,
    timeout_bars: int = 60,
    timestamps=None,
    save_csv: bool = True,
    csv_path: str | Path = "results/trade_log.csv",
    print_first_n: int = 20,
) -> dict:
    engine = TradePlanEngine(min_rr=min_rr)
    equity = float(capital)
    equity_curve = np.full(len(close), equity, dtype=np.float64)
    trades, wins, losses = [], 0, 0
    gross_profit, gross_loss = 0.0, 0.0
    tp_count, sl_count, timeout_count = 0, 0, 0
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

        raw_open = float(open_[entry_idx])
        spread_half, slippage_entry = raw_open * 0.0001, raw_open * 0.0001
        entry = raw_open + spread_half + slippage_entry if side == "LONG" else raw_open - spread_half - slippage_entry

        current_atr = float(atr[i])
        if not np.isfinite(current_atr) or current_atr <= 0:
            i += 1
            continue

        sh = float(swing_high[i]) if np.isfinite(swing_high[i]) else None
        sl = float(swing_low[i]) if np.isfinite(swing_low[i]) else None
        plan = engine.build_from_market(side=side, entry=entry, atr=current_atr, equity=equity, swing_high=sh, swing_low=sl)

        if plan.decision != TradeDecision.ACCEPTED:
            equity_curve[i] = equity
            i += 1
            continue

        exit_idx = min(entry_idx + timeout_bars, len(close) - 1)
        exit_price, exit_reason = float(close[exit_idx]), "TIMEOUT"
        active_sl = plan.stop_loss
        risk_dist = abs(plan.entry - plan.stop_loss)

        for j in range(entry_idx, exit_idx + 1):
            b_open, b_high, b_low = float(open_[j]), float(high[j]), float(low[j])
            # Strictly hold until original SL or TP (matching Triple Barrier definition)
            if side == "LONG":
                hit_sl = b_low <= plan.stop_loss
                hit_tp = b_high >= plan.take_profit
            else:
                hit_sl = b_high >= plan.stop_loss
                hit_tp = b_low <= plan.take_profit

            if hit_sl and hit_tp:
                exit_idx, exit_price, exit_reason = (j, plan.take_profit, "TP") if abs(b_open - plan.take_profit) < abs(b_open - active_sl) else (j, active_sl, "SL")
                break
            elif hit_sl:
                exit_idx, exit_price, exit_reason = j, active_sl, "SL"
                break
            elif hit_tp:
                exit_idx, exit_price, exit_reason = j, plan.take_profit, "TP"
                break

        gross_pnl = (exit_price - plan.entry) * plan.position_size if side == "LONG" else (plan.entry - exit_price) * plan.position_size
        pnl = gross_pnl - plan.estimated_costs
        equity = max(equity + pnl, 1.0)
        planned_risk = (plan.position_size * risk_dist) + plan.estimated_costs
        r_multiple = pnl / planned_risk if planned_risk > 0 else 0.0

        if exit_reason == "TP":
            tp_count += 1
        elif exit_reason == "SL":
            sl_count += 1
        else:
            timeout_count += 1

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        elif pnl < 0:
            losses += 1
            gross_loss += abs(pnl)

        trades.append({
            "trade_id": len(trades) + 1,
            "entry_idx": entry_idx,
            "exit_idx": exit_idx,
            "side": side,
            "entry": plan.entry,
            "stop_loss": plan.stop_loss,
            "take_profit": plan.take_profit,
            "pnl": pnl,
            "r_multiple": r_multiple,
            "exit_reason": exit_reason,
            "equity_after": equity,
        })
        equity_curve[i:exit_idx + 1] = equity
        i = exit_idx + 1

    total_trades = len(trades)
    win_rate = (wins / total_trades) if total_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0

    return {
        "final_equity": equity,
        "total_trades": total_trades,
        "trades_count": total_trades,
        "wins": wins,
        "losses": losses,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "timeout_count": timeout_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "trades": trades,
        "equity_curve": equity_curve,
    }
