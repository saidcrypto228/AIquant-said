from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .trade_plan import TradeDecision, TradePlanEngine, RiskModel, ConvictionRiskEngine

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
    capital: float = 10_000.0,
    risk_per_trade: float = 0.0075,
    min_rr: float = 0.5,
    timeout_bars: int = 60,
    timestamps=None,
    save_csv: bool = True,
    csv_path: str | Path = "results/trade_log.csv",
    print_first_n: int = 20,
    probs: np.ndarray | None = None,
) -> dict:
    risk_model = RiskModel(risk_per_trade=risk_per_trade)
    engine = TradePlanEngine(min_rr=min_rr, risk_model=risk_model)
    conv_engine = ConvictionRiskEngine(
        base_risk_pct=risk_per_trade,
        min_risk_pct=0.005,
        max_risk_pct=0.012,
        max_account_leverage=1.5,
        friction_pct=0.00095
    )

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

                # 0. PropGuard: проверка суточной просадки (за последние 96 баров = 24 часа)
        cutoff_bar = i - 96
        rolling_daily_pnl = sum(t["pnl"] for t in trades if t["exit_idx"] >= cutoff_bar)
        daily_loss_limit = capital * 0.030  # 3.0% ($300 на $10k счете)

        if rolling_daily_pnl <= -daily_loss_limit:
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

        # Расчет динамического риска по уверенности модели
        curr_prob = float(probs[i]) if (probs is not None and i < len(probs)) else 0.60
        est_stop_dist = max(current_atr * 2.0, entry * engine.min_safe_stop_pct)
        est_target_dist = est_stop_dist * engine.target_rr

        dynamic_risk = conv_engine.compute_conviction_risk(
            calibrated_prob=curr_prob,
            stop_dist_pct=est_stop_dist / entry,
            target_dist_pct=est_target_dist / entry
        )
        if dynamic_risk <= 0:
            equity_curve[i] = equity
            i += 1
            continue

        engine.risk_model = RiskModel(risk_per_trade=dynamic_risk)
        plan = engine.build_from_market(side=side, entry=entry, atr=current_atr, equity=equity, swing_high=sh, swing_low=sl)

        if plan.decision != TradeDecision.ACCEPTED:
            equity_curve[i] = equity
            i += 1
            continue

        exit_idx = min(entry_idx + timeout_bars, len(close) - 1)
        exit_price, exit_reason = float(close[exit_idx]), 'TIMEOUT'
        active_sl = float(plan.stop_loss)
        risk_dist = abs(plan.entry - plan.stop_loss)
        init_tp_dist = abs(plan.take_profit - plan.entry)
        stagnation_bar = 12
        min_tp_buffer = max(risk_dist * 0.25, plan.entry * 0.0025)

        for j in range(entry_idx, exit_idx + 1):
            b_open, b_high, b_low, b_close = float(open_[j]), float(high[j]), float(low[j]), float(close[j])
            bars_held = j - entry_idx

            # Трейлинг стопа
            if side == 'LONG':
                unrealized_r = (b_high - plan.entry) / risk_dist if risk_dist > 0 else 0.0
                if unrealized_r >= 1.50:
                    active_sl = max(active_sl, plan.entry + (1.00 * risk_dist))
                elif unrealized_r >= 1.00:
                    active_sl = max(active_sl, plan.entry + (0.25 * risk_dist))
            else:
                unrealized_r = (plan.entry - b_low) / risk_dist if risk_dist > 0 else 0.0
                if unrealized_r >= 1.50:
                    active_sl = min(active_sl, plan.entry - (1.00 * risk_dist))
                elif unrealized_r >= 1.00:
                    active_sl = min(active_sl, plan.entry - (0.25 * risk_dist))

            # Decaying TP
            if bars_held < stagnation_bar:
                curr_tp_dist = init_tp_dist
            else:
                progress = (bars_held - stagnation_bar) / max(float(timeout_bars - stagnation_bar), 1.0)
                decay_factor = max(0.0, 1.0 - (progress ** 2))
                curr_tp_dist = max(init_tp_dist * decay_factor, min_tp_buffer)

            curr_tp = plan.entry + curr_tp_dist if side == 'LONG' else plan.entry - curr_tp_dist
            is_decayed = (bars_held >= stagnation_bar)

            # Time-Decay SL
            if bars_held >= 12:
                curr_unrealized_r = (b_close - plan.entry) / risk_dist if side == 'LONG' else (plan.entry - b_close) / risk_dist
                if curr_unrealized_r <= -0.25:
                    exit_idx, exit_price, exit_reason = j, b_close, 'TIME_DECAY_SL'
                    break

            if side == 'LONG':
                hit_sl = b_low <= active_sl
                hit_tp = b_high >= curr_tp
            else:
                hit_sl = b_high >= active_sl
                hit_tp = b_low <= curr_tp

            tp_label = 'DECAY_TP' if is_decayed else 'TP'
            if hit_sl and hit_tp:
                exit_idx, exit_price, exit_reason = (j, curr_tp, tp_label) if abs(b_open - curr_tp) < abs(b_open - active_sl) else (j, active_sl, 'SL')
                break
            elif hit_sl:
                exit_idx, exit_price, exit_reason = j, active_sl, 'SL'
                break
            elif hit_tp:
                exit_idx, exit_price, exit_reason = j, curr_tp, tp_label
                break

        gross_pnl = (exit_price - plan.entry) * plan.position_size if side == "LONG" else (plan.entry - exit_price) * plan.position_size
        pnl = gross_pnl - plan.estimated_costs
        equity = max(equity + pnl, 1.0)
        planned_risk = (plan.position_size * risk_dist) + plan.estimated_costs
        r_multiple = pnl / planned_risk if planned_risk > 0 else 0.0

        if exit_reason in ("TP", "DECAY_TP"):
            tp_count += 1
        elif exit_reason in ("SL", "TIME_DECAY_SL"):
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
            "exit": exit_price,
            "exit_reason": exit_reason,
            "pnl": pnl,
            "r_multiple": r_multiple,
            "equity_after": equity,
            "prob": curr_prob
        })

        cooldown = 4
        next_i = min(exit_idx + 1 + cooldown, len(close) - 1)
        equity_curve[i:next_i] = equity
        i = next_i

    equity_curve[i:] = equity
    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)

    if total_trades > 0:
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        win_rate = (wins / total_trades) * 100.0
    else:
        pf, win_rate = 0.0, 0.0

    eq_series = pd.Series(equity_curve[test_idx])
    roll_max = eq_series.cummax()
    drawdowns = (eq_series - roll_max) / roll_max
    max_dd = abs(float(drawdowns.min())) * 100.0
    ret_pct = ((equity - capital) / capital) * 100.0

    returns = eq_series.pct_change().dropna()
    sharpe = float(np.sqrt(365 * 24 * 4) * (returns.mean() / returns.std())) if len(returns) > 1 and returns.std() > 0 else 0.0
    calmar = (ret_pct / max_dd) if max_dd > 0 else 0.0

    if save_csv and total_trades > 0:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_trades.to_csv(csv_path, index=False)

    return {
        "final": equity,
        "final_equity": equity,
        "total_return_pct": ret_pct,
        "ret": ret_pct,
        "max_drawdown_pct": max_dd,
        "max_dd": max_dd,
        "profit_factor": pf,
        "pf": pf,
        "sharpe_ratio": sharpe,
        "sharpe": sharpe,
        "calmar_ratio": calmar,
        "calmar": calmar,
        "trades": trades,              # список словарей сделок (для итерации)
        "total_trades": total_trades,  # целочисленный счетчик
        "trade_count": total_trades,
        "n_trades": total_trades,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "timeout_count": timeout_count,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "equity_curve": equity_curve,
        "trades_df": df_trades,
        "long_thresh": 0.58,
        "short_thresh": 0.58
    }
