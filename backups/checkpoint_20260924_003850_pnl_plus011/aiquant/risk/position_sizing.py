"""
aiquant/risk/position_sizing.py
=================================
Professional position sizing using Kelly Criterion and variants.

Implements:
  - Full Kelly Criterion
  - Fractional Kelly (Half-Kelly default — safer for live trading)
  - Volatility-adjusted Kelly
  - Rolling Kelly (adapts to recent win/loss history)
  - Fixed Fractional (fallback)
  - Optimal f (Ralph Vince)

References:
  Kelly, J.L. (1956). A New Interpretation of Information Rate.
  Thorp, E.O. (2006). The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market.
  Vince, R. (1992). The Mathematics of Money Management.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KellyCriterion:
    """
    Full-featured Kelly Criterion position sizer.
    Supports full Kelly, fractional Kelly, and volatility-adjusted Kelly.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.5,    # Half-Kelly by default
        max_position_pct: float = 0.25,  # Hard cap: never exceed 25% of portfolio
        min_position_pct: float = 0.01,  # Minimum meaningful position
        lookback_trades: int = 50,       # Rolling window for win/loss estimation
    ):
        self.kelly_fraction = kelly_fraction
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.lookback_trades = lookback_trades

    def full_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Classic Kelly formula: f* = (p * b - q) / b
        where p = win probability, q = 1 - p, b = win/loss ratio.
        """
        if avg_loss == 0 or avg_win == 0:
            return 0.0
        b = abs(avg_win / avg_loss)  # Win/loss ratio
        p = win_rate
        q = 1 - p
        kelly = (p * b - q) / b
        return max(0.0, kelly)

    def fractional_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Apply fractional Kelly multiplier to full Kelly."""
        fk = self.full_kelly(win_rate, avg_win, avg_loss) * self.kelly_fraction
        return np.clip(fk, self.min_position_pct, self.max_position_pct)

    def volatility_adjusted_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_vol: float,
        baseline_vol: float,
    ) -> float:
        """
        Scale Kelly fraction inversely with volatility.
        Higher current vol → smaller position.
        """
        base_kelly = self.fractional_kelly(win_rate, avg_win, avg_loss)
        if baseline_vol == 0:
            return base_kelly
        vol_scalar = baseline_vol / max(current_vol, 1e-8)
        vol_scalar = np.clip(vol_scalar, 0.25, 2.0)  # Limit scaling range
        adjusted = base_kelly * vol_scalar
        return np.clip(adjusted, self.min_position_pct, self.max_position_pct)

    def rolling_kelly(self, trade_log: pd.DataFrame, current_vol: float = None, baseline_vol: float = None) -> float:
        """
        Compute Kelly from recent trade history (rolling window).
        Adapts position sizing to recent strategy performance.
        """
        if trade_log is None or len(trade_log) < 10:
            logger.warning("Insufficient trade history for rolling Kelly. Using minimum position.")
            return self.min_position_pct

        recent = trade_log.tail(self.lookback_trades)
        pnl = recent['pnl_net'] if 'pnl_net' in recent.columns else recent.get('pnl', pd.Series())

        if pnl.empty:
            return self.min_position_pct

        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = len(wins) / len(pnl)
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 1e-8

        if current_vol is not None and baseline_vol is not None:
            return self.volatility_adjusted_kelly(win_rate, avg_win, avg_loss, current_vol, baseline_vol)
        return self.fractional_kelly(win_rate, avg_win, avg_loss)

    def compute_position_size(
        self,
        portfolio_value: float,
        price: float,
        kelly_fraction_override: Optional[float] = None,
        trade_log: Optional[pd.DataFrame] = None,
        current_vol: Optional[float] = None,
        baseline_vol: Optional[float] = None,
    ) -> dict:
        """
        Compute the actual position size in USD and units.
        Returns a dict with size details for order placement.
        """
        if kelly_fraction_override is not None:
            fraction = np.clip(kelly_fraction_override, self.min_position_pct, self.max_position_pct)
        elif trade_log is not None and len(trade_log) >= 10:
            fraction = self.rolling_kelly(trade_log, current_vol, baseline_vol)
        else:
            fraction = self.kelly_fraction * 0.5  # Conservative default

        position_usd = portfolio_value * fraction
        position_units = position_usd / price

        return {
            'kelly_fraction_used': fraction,
            'position_usd': position_usd,
            'position_units': position_units,
            'portfolio_value': portfolio_value,
            'price': price,
        }


class OptimalF:
    """
    Ralph Vince's Optimal f — maximises geometric growth rate.
    More aggressive than Kelly; use with caution.
    """

    def __init__(self, max_fraction: float = 0.25):
        self.max_fraction = max_fraction

    def compute(self, trade_returns: pd.Series) -> float:
        """
        Find the f value that maximises the Terminal Wealth Relative (TWR).
        """
        if len(trade_returns) < 10:
            return 0.01

        worst_loss = abs(trade_returns.min())
        if worst_loss == 0:
            return 0.01

        def neg_twr(f):
            if f <= 0:
                return 0
            hpr = 1 + f * trade_returns / worst_loss
            hpr = hpr.clip(lower=0.001)
            twr = hpr.prod()
            return -twr  # Minimise negative TWR

        result = minimize_scalar(neg_twr, bounds=(0.001, self.max_fraction), method='bounded')
        return result.x


class RiskManager:
    """
    Portfolio-level risk manager.
    Enforces drawdown limits, position limits, and daily loss limits.
    """

    def __init__(
        self,
        max_drawdown_limit: float = 0.15,   # 15% max drawdown before halting
        daily_loss_limit: float = 0.03,      # 3% daily loss limit
        max_open_positions: int = 1,         # BTC is single-asset
        max_position_pct: float = 0.25,
        kelly_fraction: float = 0.5,
    ):
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_loss_limit = daily_loss_limit
        self.max_open_positions = max_open_positions
        self.max_position_pct = max_position_pct
        self.kelly = KellyCriterion(kelly_fraction=kelly_fraction, max_position_pct=max_position_pct)

        self.peak_value: float = 0.0
        self.daily_start_value: float = 0.0
        self.is_halted: bool = False

    def update(self, portfolio_value: float):
        """Update risk state with current portfolio value."""
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value
        if self.daily_start_value == 0:
            self.daily_start_value = portfolio_value

    def current_drawdown(self, portfolio_value: float) -> float:
        """Current drawdown from peak."""
        if self.peak_value == 0:
            return 0.0
        return (self.peak_value - portfolio_value) / self.peak_value

    def daily_loss(self, portfolio_value: float) -> float:
        """Loss since start of day."""
        if self.daily_start_value == 0:
            return 0.0
        return (self.daily_start_value - portfolio_value) / self.daily_start_value

    def can_trade(self, portfolio_value: float) -> dict:
        """
        Check if trading is allowed given current risk state.
        Returns a dict with allow flag and reason.
        """
        dd = self.current_drawdown(portfolio_value)
        dl = self.daily_loss(portfolio_value)

        if dd >= self.max_drawdown_limit:
            self.is_halted = True
            return {'allow': False, 'reason': f'Max drawdown breached: {dd:.2%}'}

        if dl >= self.daily_loss_limit:
            return {'allow': False, 'reason': f'Daily loss limit breached: {dl:.2%}'}

        return {'allow': True, 'reason': 'OK', 'drawdown': dd, 'daily_loss': dl}

    def get_position_size(
        self,
        portfolio_value: float,
        price: float,
        trade_log: Optional[pd.DataFrame] = None,
        current_vol: Optional[float] = None,
        baseline_vol: Optional[float] = None,
    ) -> dict:
        """
        Get Kelly-sized position with risk checks applied.
        """
        risk_check = self.can_trade(portfolio_value)
        if not risk_check['allow']:
            logger.warning(f"Trade blocked: {risk_check['reason']}")
            return {'position_usd': 0, 'position_units': 0, 'blocked': True, 'reason': risk_check['reason']}

        sizing = self.kelly.compute_position_size(
            portfolio_value=portfolio_value,
            price=price,
            trade_log=trade_log,
            current_vol=current_vol,
            baseline_vol=baseline_vol,
        )
        sizing['blocked'] = False
        sizing['drawdown'] = risk_check.get('drawdown', 0)
        return sizing

    def reset_daily(self, portfolio_value: float):
        """Reset daily loss counter (call at start of each trading day)."""
        self.daily_start_value = portfolio_value
        logger.info(f"Daily risk reset. Portfolio: ${portfolio_value:,.2f}")
