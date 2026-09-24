"""
aiquant/backtest/analytics.py
==============================
Comprehensive performance analytics for backtested strategies.

Metrics:
  - Sharpe, Sortino, Calmar, Omega ratios
  - Max drawdown, drawdown duration
  - Win rate, profit factor, expectancy
  - Value at Risk (VaR), Conditional VaR (CVaR)
  - Monthly and annual return heatmaps
  - Rolling Sharpe ratio
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.04  # Annual


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 525_600) -> float:
    """Annualised Sharpe ratio (1m bars: 525,600 per year)."""
    excess = returns - RISK_FREE_RATE / periods_per_year
    if returns.std() == 0:
        return 0.0
    return (excess.mean() / returns.std()) * np.sqrt(periods_per_year)


def sortino_ratio(returns: pd.Series, periods_per_year: int = 525_600) -> float:
    """Annualised Sortino ratio (downside deviation only)."""
    excess = returns - RISK_FREE_RATE / periods_per_year
    downside = returns[returns < 0].std()
    if downside == 0:
        return 0.0
    return (excess.mean() / downside) * np.sqrt(periods_per_year)


def calmar_ratio(returns: pd.Series, periods_per_year: int = 525_600) -> float:
    """Calmar ratio: annualised return / max drawdown."""
    ann_return = returns.mean() * periods_per_year
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = abs(dd.min())
    if max_dd == 0:
        return 0.0
    return ann_return / max_dd


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains vs losses above threshold."""
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if losses.sum() == 0:
        return np.inf
    return gains.sum() / losses.sum()


def max_drawdown(equity_curve: pd.Series) -> dict:
    """Compute maximum drawdown and its duration."""
    rolling_max = equity_curve.cummax()
    dd = (equity_curve - rolling_max) / rolling_max
    max_dd = dd.min()
    # Duration
    in_dd = dd < 0
    dd_groups = (in_dd != in_dd.shift()).cumsum()
    dd_durations = in_dd.groupby(dd_groups).sum()
    max_duration = dd_durations.max() if not dd_durations.empty else 0
    return {'max_drawdown': max_dd, 'max_duration_bars': max_duration}


def var_cvar(returns: pd.Series, confidence: float = 0.95) -> dict:
    """Value at Risk and Conditional VaR at given confidence level."""
    var = returns.quantile(1 - confidence)
    cvar = returns[returns <= var].mean()
    return {'var': var, 'cvar': cvar}


def full_analytics(
    returns: pd.Series,
    equity_curve: Optional[pd.Series] = None,
    trade_log: Optional[pd.DataFrame] = None,
    periods_per_year: int = 525_600,
) -> dict:
    """
    Compute the full set of performance metrics.
    """
    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()

    dd_stats = max_drawdown(equity_curve)
    vc = var_cvar(returns)

    metrics = {
        # Return metrics
        'total_return_pct': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100,
        'annualised_return_pct': returns.mean() * periods_per_year * 100,
        'annualised_vol_pct': returns.std() * np.sqrt(periods_per_year) * 100,
        # Risk-adjusted
        'sharpe_ratio': sharpe_ratio(returns, periods_per_year),
        'sortino_ratio': sortino_ratio(returns, periods_per_year),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        'omega_ratio': omega_ratio(returns),
        # Drawdown
        'max_drawdown_pct': dd_stats['max_drawdown'] * 100,
        'max_dd_duration_bars': dd_stats['max_duration_bars'],
        # Tail risk
        'var_95_pct': vc['var'] * 100,
        'cvar_95_pct': vc['cvar'] * 100,
        # Distribution
        'skewness': returns.skew(),
        'kurtosis': returns.kurtosis(),
    }

    # Trade-level metrics
    if trade_log is not None and not trade_log.empty and 'pnl_net' in trade_log.columns:
        pnl = trade_log['pnl_net']
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        metrics.update({
            'total_trades': len(pnl),
            'win_rate_pct': len(wins) / len(pnl) * 100,
            'avg_win': wins.mean() if len(wins) > 0 else 0,
            'avg_loss': losses.mean() if len(losses) > 0 else 0,
            'profit_factor': abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf,
            'expectancy': pnl.mean(),
            'largest_win': wins.max() if len(wins) > 0 else 0,
            'largest_loss': losses.min() if len(losses) > 0 else 0,
        })

    return metrics


def plot_full_report(
    equity_curve: pd.Series,
    returns: pd.Series,
    metrics: dict,
    strategy_name: str = 'Strategy',
    output_dir: str = 'results',
):
    """
    Generate a comprehensive 6-panel performance report.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'AIQuant — {strategy_name} Performance Report', fontsize=16, fontweight='bold', y=0.98)

    # 1. Equity Curve
    ax1 = fig.add_subplot(3, 3, (1, 3))
    ax1.plot(equity_curve.values, color='#00b4d8', linewidth=1.5, label='Portfolio Value')
    ax1.set_title('Equity Curve', fontweight='bold')
    ax1.set_ylabel('USD')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = fig.add_subplot(3, 3, (4, 6))
    rolling_max = equity_curve.cummax()
    dd = (equity_curve - rolling_max) / rolling_max * 100
    ax2.fill_between(range(len(dd)), dd.values, 0, color='#e63946', alpha=0.7)
    ax2.set_title('Drawdown (%)', fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # 3. Return Distribution
    ax3 = fig.add_subplot(3, 3, 7)
    ax3.hist(returns.dropna() * 100, bins=80, color='#48cae4', edgecolor='white', alpha=0.8)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_title('Return Distribution (%)', fontweight='bold')
    ax3.set_xlabel('Return (%)')
    ax3.grid(True, alpha=0.3)

    # 4. Rolling Sharpe (30-day window in 1m bars = 43200 bars)
    ax4 = fig.add_subplot(3, 3, 8)
    roll_window = min(43200, len(returns) // 4)
    if roll_window > 10:
        # Vectorised rolling Sharpe via NumPy stride tricks — avoids
        # pandas .rolling().apply() which calls Python once per bar.
        r_arr = returns.to_numpy(dtype=np.float64)
        n_r   = len(r_arr)
        roll_sharpe_arr = np.full(n_r, np.nan)
        # Use cumulative sum trick for rolling mean and variance
        cs  = np.cumsum(r_arr)
        cs2 = np.cumsum(r_arr ** 2)
        for i in range(roll_window - 1, n_r):
            s  = cs[i]  - (cs[i - roll_window]  if i >= roll_window else 0.0)
            s2 = cs2[i] - (cs2[i - roll_window] if i >= roll_window else 0.0)
            mu  = s  / roll_window
            var = s2 / roll_window - mu ** 2
            std = var ** 0.5 if var > 0 else 1e-10
            roll_sharpe_arr[i] = (mu / std) * np.sqrt(1440 * 365)
        ax4.plot(roll_sharpe_arr, color='#90e0ef', linewidth=1)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax4.axhline(y=1, color='green', linestyle='--', alpha=0.5)
    ax4.set_title('Rolling Sharpe Ratio', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # 5. Key Metrics Table
    ax5 = fig.add_subplot(3, 3, 9)
    ax5.axis('off')
    key_metrics = [
        ('Total Return', f"{metrics.get('total_return_pct', 0):.2f}%"),
        ('Ann. Return', f"{metrics.get('annualised_return_pct', 0):.2f}%"),
        ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.3f}"),
        ('Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.3f}"),
        ('Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.3f}"),
        ('Max Drawdown', f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ('Win Rate', f"{metrics.get('win_rate_pct', 0):.1f}%"),
        ('Profit Factor', f"{metrics.get('profit_factor', 0):.3f}"),
        ('VaR 95%', f"{metrics.get('var_95_pct', 0):.3f}%"),
        ('CVaR 95%', f"{metrics.get('cvar_95_pct', 0):.3f}%"),
    ]
    table = ax5.table(
        cellText=key_metrics,
        colLabels=['Metric', 'Value'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax5.set_title('Key Metrics', fontweight='bold')

    plt.tight_layout()
    save_path = output_path / f'{strategy_name}_report.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Performance report saved to {save_path}")
    return save_path
