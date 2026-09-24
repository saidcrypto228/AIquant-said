"""
aiquant/backtest/engine.py
===========================
Professional Backtrader backtesting engine for 1m BTCUSD.

Features:
  - Signal-driven strategy (pre-computed signals injected as a data feed)
  - Realistic fee model (Hyperliquid taker: 0.035%)
  - Slippage model (market impact)
  - Full trade log with entry/exit prices, PnL, hold time
  - Walk-forward analysis support
  - Comprehensive performance analytics
"""

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom Backtrader Data Feed (OHLCV + pre-computed signal)
# ---------------------------------------------------------------------------

class BTCSignalData(bt.feeds.PandasData):
    """
    Extended Backtrader data feed that includes a pre-computed signal column.
    """
    lines = ('signal',)
    params = (
        ('signal', -1),  # Column index or name
        ('openinterest', -1),
    )


# ---------------------------------------------------------------------------
# Backtrader Strategy: Signal Execution
# ---------------------------------------------------------------------------

class SignalExecutionStrategy(bt.Strategy):
    """
    Executes pre-computed signals from the feature/strategy pipeline.
    Handles position sizing, stop-loss, take-profit, and max hold time.
    """

    params = (
        ('stop_loss_pct', 0.01),
        ('take_profit_pct', 0.03),
        ('max_hold_bars', 60),
        ('position_size_pct', 0.20),  # % of portfolio per trade (Kelly-adjusted externally)
        ('verbose', True),
    )

    def __init__(self):
        self.signal = self.data.signal
        self.order = None
        self.entry_price = None
        self.entry_bar = None
        self.trade_log = []

    def log(self, txt, dt=None):
        if self.params.verbose:
            dt = dt or self.datas[0].datetime.datetime(0)
            logger.debug(f'{dt.isoformat()} | {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                self.log(f'BUY  | Price: {order.executed.price:.2f} | Size: {order.executed.size:.6f}')
            elif order.issell():
                self.log(f'SELL | Price: {order.executed.price:.2f} | Size: {order.executed.size:.6f}')
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order {order.status}')
        self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.trade_log.append({
                'open_date': bt.num2date(trade.dtopen),
                'close_date': bt.num2date(trade.dtclose),
                'pnl': trade.pnl,
                'pnl_net': trade.pnlcomm,
                'size': trade.size,
                'price_open': trade.price,
                'commission': trade.commission,
            })

    def next(self):
        if self.order:
            return

        current_signal = self.signal[0]
        portfolio_value = self.broker.getvalue()
        price = self.data.close[0]
        size = (portfolio_value * self.params.position_size_pct) / price

        # Check stop-loss and take-profit for open positions
        if self.position:
            bars_held = len(self) - self.entry_bar
            pnl_pct = (price - self.entry_price) / self.entry_price

            if self.position.size > 0:  # Long position
                if (pnl_pct <= -self.params.stop_loss_pct
                        or pnl_pct >= self.params.take_profit_pct
                        or bars_held >= self.params.max_hold_bars):
                    self.order = self.close()
                    return
            elif self.position.size < 0:  # Short position
                if (-pnl_pct <= -self.params.stop_loss_pct
                        or -pnl_pct >= self.params.take_profit_pct
                        or bars_held >= self.params.max_hold_bars):
                    self.order = self.close()
                    return

        # Entry logic
        if not self.position:
            if current_signal == 1:
                self.order = self.buy(size=size)
                self.entry_price = price
                self.entry_bar = len(self)
            elif current_signal == -1:
                self.order = self.sell(size=size)
                self.entry_price = price
                self.entry_bar = len(self)
        else:
            # Signal reversal: close and reverse
            if self.position.size > 0 and current_signal == -1:
                self.order = self.close()
            elif self.position.size < 0 and current_signal == 1:
                self.order = self.close()


# ---------------------------------------------------------------------------
# Backtest Runner
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Orchestrates Backtrader backtests with full analytics.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fees: float = 0.00035,       # Hyperliquid taker fee
        slippage: float = 0.0001,
        stop_loss_pct: float = 0.01,
        take_profit_pct: float = 0.03,
        max_hold_bars: int = 60,
        position_size_pct: float = 0.20,
        output_dir: str = 'results',
    ):
        self.initial_capital = initial_capital
        self.fees = fees
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_bars = max_hold_bars
        self.position_size_pct = position_size_pct
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        strategy_name: str = 'strategy',
        plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a backtest given OHLCV DataFrame and pre-computed signals.
        """
        logger.info(f"Running backtest: {strategy_name}")

        # Merge signals into DataFrame
        df_bt = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df_bt['signal'] = signals.reindex(df_bt.index).fillna(0)

        # Ensure DatetimeIndex
        if not isinstance(df_bt.index, pd.DatetimeIndex):
            df_bt.index = pd.to_datetime(df_bt.index)

        # Backtrader cerebro setup
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(self.initial_capital)
        cerebro.broker.setcommission(commission=self.fees)
        cerebro.broker.set_slippage_perc(self.slippage)

        # Add data feed
        data_feed = BTCSignalData(
            dataname=df_bt,
            datetime=None,
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            signal='signal',
        )
        cerebro.adddata(data_feed)

        # Add strategy
        cerebro.addstrategy(
            SignalExecutionStrategy,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            max_hold_bars=self.max_hold_bars,
            position_size_pct=self.position_size_pct,
            verbose=False,
        )

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.04, annualize=True)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return', timeframe=bt.TimeFrame.Days)

        # Run
        start_value = cerebro.broker.getvalue()
        results = cerebro.run()
        end_value = cerebro.broker.getvalue()
        strat = results[0]

        # Extract analytics
        sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio', None)
        drawdown = strat.analyzers.drawdown.get_analysis()
        trade_analysis = strat.analyzers.trades.get_analysis()
        sqn = strat.analyzers.sqn.get_analysis().get('sqn', None)
        time_return = strat.analyzers.time_return.get_analysis()

        # Build equity curve from time returns
        if time_return:
            equity = pd.Series(time_return).sort_index()
            equity_curve = (1 + equity).cumprod() * self.initial_capital
        else:
            equity_curve = pd.Series([self.initial_capital, end_value])

        # Trade log from strategy
        trade_log = pd.DataFrame(strat.trade_log)

        # Compute metrics
        total_return = (end_value - start_value) / start_value
        max_dd = drawdown.get('max', {}).get('drawdown', 0) / 100

        total_trades = trade_analysis.get('total', {}).get('closed', 0)
        won_trades = trade_analysis.get('won', {}).get('total', 0)
        lost_trades = trade_analysis.get('lost', {}).get('total', 0)
        win_rate = won_trades / total_trades if total_trades > 0 else 0

        avg_win = trade_analysis.get('won', {}).get('pnl', {}).get('average', 0)
        avg_loss = trade_analysis.get('lost', {}).get('pnl', {}).get('average', 0)
        profit_factor = abs(avg_win * won_trades / (avg_loss * lost_trades)) if (avg_loss and lost_trades) else np.nan

        metrics = {
            'strategy': strategy_name,
            'initial_capital': start_value,
            'final_value': end_value,
            'total_return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'sqn': sqn,
            'total_trades': total_trades,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
        }

        # Print summary
        logger.info("=" * 55)
        logger.info(f"  BACKTEST RESULTS: {strategy_name}")
        logger.info("=" * 55)
        for k, v in metrics.items():
            if isinstance(v, float):
                logger.info(f"  {k:<28}: {v:.4f}")
            else:
                logger.info(f"  {k:<28}: {v}")
        logger.info("=" * 55)

        # Save trade log
        if not trade_log.empty:
            trade_log.to_csv(self.output_dir / f'{strategy_name}_trades.csv', index=False)

        # Plot
        if plot:
            self._plot_equity(equity_curve, metrics, strategy_name)

        return {
            'metrics': metrics,
            'equity_curve': equity_curve,
            'trade_log': trade_log,
        }

    def _plot_equity(self, equity_curve: pd.Series, metrics: dict, name: str):
        """Generate and save equity curve plot."""
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # Equity curve
        axes[0].plot(equity_curve.values, color='#00b4d8', linewidth=1.5)
        axes[0].set_title(f'{name} — Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value (USD)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=metrics['initial_capital'], color='gray', linestyle='--', alpha=0.5)

        # Drawdown
        equity_arr = equity_curve.values
        rolling_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - rolling_max) / rolling_max * 100
        axes[1].fill_between(range(len(drawdown)), drawdown, 0, color='#e63946', alpha=0.6)
        axes[1].set_title('Drawdown (%)', fontsize=12)
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)

        # Annotation
        ann = (
            f"Return: {metrics['total_return_pct']:.2f}%  |  "
            f"Sharpe: {metrics['sharpe_ratio']:.2f}  |  "
            f"Max DD: {metrics['max_drawdown_pct']:.2f}%  |  "
            f"Win Rate: {metrics['win_rate_pct']:.1f}%  |  "
            f"Trades: {metrics['total_trades']}"
        )
        fig.suptitle(ann, fontsize=11, y=0.02)
        plt.tight_layout()

        path = self.output_dir / f'{name}_equity.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Equity curve saved to {path}")


# ---------------------------------------------------------------------------
# Walk-Forward Analysis
# ---------------------------------------------------------------------------

class WalkForwardAnalysis:
    """
    Walk-forward backtesting to prevent overfitting.
    Splits data into in-sample (training) and out-of-sample (testing) windows.
    """

    def __init__(
        self,
        train_bars: int = 10_000,
        test_bars: int = 2_000,
        step_bars: int = 2_000,
        engine_kwargs: Optional[dict] = None,
    ):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.engine_kwargs = engine_kwargs or {}

    def run(
        self,
        df: pd.DataFrame,
        signal_generator,  # Callable: df -> pd.Series
        strategy_name: str = 'wfa_strategy',
    ) -> pd.DataFrame:
        """
        Run walk-forward analysis.
        Returns a DataFrame of out-of-sample performance metrics per window.
        """
        engine = BacktestEngine(**self.engine_kwargs)
        n = len(df)
        results = []
        window = 0

        start = 0
        while start + self.train_bars + self.test_bars <= n:
            train_end = start + self.train_bars
            test_end = train_end + self.test_bars

            df_train = df.iloc[start:train_end]
            df_test = df.iloc[train_end:test_end]

            logger.info(
                f"WFA Window {window+1}: "
                f"Train [{df_train.index[0]} → {df_train.index[-1]}] "
                f"Test [{df_test.index[0]} → {df_test.index[-1]}]"
            )

            # Generate signals on test set (using model trained on train set)
            try:
                signals_test = signal_generator(df_train, df_test)
                result = engine.run(df_test, signals_test, f'{strategy_name}_wfa_{window}', plot=False)
                result['metrics']['window'] = window
                result['metrics']['test_start'] = str(df_test.index[0])
                result['metrics']['test_end'] = str(df_test.index[-1])
                results.append(result['metrics'])
            except Exception as e:
                logger.error(f"WFA window {window} failed: {e}")

            start += self.step_bars
            window += 1

        df_results = pd.DataFrame(results)
        logger.info(f"\nWalk-Forward Summary:\n{df_results.describe()}")
        return df_results
