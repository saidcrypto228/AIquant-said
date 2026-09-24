"""
aiquant/execution/live_trader.py
=================================
Live trading orchestrator for AIQuant.

Workflow:
  1. Fetch latest 1m candles from Hyperliquid public API (no auth needed)
  2. Build features on the rolling window
  3. Generate signal via StrategyEnsemble
  4. Size position with Kelly criterion
  5. Execute on Hyperliquid mainnet via HyperliquidPaperTrader

Data source : Hyperliquid public candleSnapshot API
Execution   : Hyperliquid mainnet (requires HYPERLIQUID_PRIVATE_KEY in .env)
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LiveTradingOrchestrator:
    """
    Orchestrates the full live trading loop:
      fetch → features → signal → size → execute

    Parameters
    ----------
    pair              : Trading pair, e.g. 'BTCUSDT'
    coin              : Hyperliquid coin name, e.g. 'BTC'
    initial_capital   : Starting capital in USD
    kelly_fraction    : Kelly fraction for position sizing (default 0.5)
    poll_interval_sec : Seconds between each trading loop tick
    log_dir           : Directory for trade logs
    feature_window    : Number of recent bars to use for feature computation
    """

    def __init__(
        self,
        pair:              str   = "BTCUSDT",
        coin:              str   = "BTC",
        initial_capital:   float = 10_000.0,
        kelly_fraction:    float = 0.5,
        poll_interval_sec: float = 60.0,
        log_dir:           str   = "logs/live_trading",
        feature_window:    int   = 500,
    ):
        self.pair              = pair.upper()
        self.coin              = coin.upper()
        self.initial_capital   = initial_capital
        self.kelly_fraction    = kelly_fraction
        self.poll_interval_sec = poll_interval_sec
        self.log_dir           = Path(log_dir)
        self.feature_window    = feature_window

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Lazy imports — only load heavy modules when start() is called
        self._trader   = None
        self._ensemble = None
        self._risk     = None

    # ── Public API ────────────────────────────────────────────────────────

    def start(self, max_iterations: Optional[int] = None):
        """
        Start the live trading loop.
        Blocks until interrupted (Ctrl+C) or max_iterations reached.
        """
        self._init_components()
        logger.info(
            f"Live trading started | {self.pair} | "
            f"Capital: ${self.initial_capital:,.2f} | "
            f"Poll: {self.poll_interval_sec}s"
        )

        iteration = 0
        while True:
            try:
                self._tick()
                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max_iterations={max_iterations}. Stopping.")
                    break
                time.sleep(self.poll_interval_sec)
            except KeyboardInterrupt:
                logger.info("Live trading stopped by user.")
                break
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)
                time.sleep(self.poll_interval_sec)

    # ── Internal ──────────────────────────────────────────────────────────

    def _init_components(self):
        """Initialise trader, strategy ensemble, and risk manager."""
        from .hyperliquid_trader import HyperliquidPaperTrader
        from ..strategies.ensemble import StrategyEnsemble
        from ..risk.position_sizing import RiskManager

        self._trader = HyperliquidPaperTrader(
            use_testnet=False,   # mainnet — set HYPERLIQUID_PRIVATE_KEY in .env
            log_dir=str(self.log_dir),
        )
        connected = self._trader.connect()
        if not connected:
            raise RuntimeError(
                "Could not connect to Hyperliquid. "
                "Ensure HYPERLIQUID_PRIVATE_KEY is set in your .env file."
            )

        self._ensemble = StrategyEnsemble()
        self._risk     = RiskManager(initial_capital=self.initial_capital)
        logger.info("All components initialised.")

    def _tick(self):
        """Single trading loop iteration."""
        from ..data.fetcher import fetch_hyperliquid_candles
        from ..features import build_full_feature_set

        # 1. Fetch latest bars from Hyperliquid (no auth needed)
        df_raw = fetch_hyperliquid_candles(
            pair=self.pair,
            n_bars=self.feature_window,
            interval="1m",
        )
        if len(df_raw) < 50:
            logger.warning(f"Insufficient bars ({len(df_raw)}). Skipping tick.")
            return

        # 2. Build features
        df_feat = build_full_feature_set(df_raw)
        df_feat = df_feat.dropna()
        if len(df_feat) < 10:
            logger.warning("Too many NaN rows after feature build. Skipping tick.")
            return

        # 3. Generate signal
        signals = self._ensemble.generate_signals(df_feat)
        latest_signal = int(signals["final_signal"].iloc[-1])

        # 4. Compute Kelly-sized position
        account   = self._trader.get_account_state()
        portfolio = account.get("account_value", self.initial_capital)

        close_arr  = df_feat["close"].to_numpy(dtype=np.float64)
        log_ret    = np.diff(np.log(close_arr[-21:]))
        vol_20     = float(np.std(log_ret) * np.sqrt(1440)) if len(log_ret) > 1 else 0.02
        size_usd   = portfolio * self.kelly_fraction * max(0.05, 1 - vol_20 * 5)
        size_usd   = min(size_usd, portfolio * 0.25)  # max 25% per trade

        # 5. Execute
        current_pos = self._get_current_position()
        ts = datetime.utcnow().strftime("%H:%M:%S")

        if latest_signal == 1 and current_pos <= 0:
            if current_pos < 0:
                self._trader.close_position(self.coin)
            self._trader.place_market_order(self.coin, "long", size_usd)
            logger.info(f"[{ts}] LONG  {self.coin} | ${size_usd:,.0f} | Portfolio: ${portfolio:,.2f}")

        elif latest_signal == -1 and current_pos >= 0:
            if current_pos > 0:
                self._trader.close_position(self.coin)
            self._trader.place_market_order(self.coin, "short", size_usd)
            logger.info(f"[{ts}] SHORT {self.coin} | ${size_usd:,.0f} | Portfolio: ${portfolio:,.2f}")

        elif latest_signal == 0 and current_pos != 0:
            self._trader.close_position(self.coin)
            logger.info(f"[{ts}] FLAT  {self.coin} | Closed position | Portfolio: ${portfolio:,.2f}")

        else:
            logger.info(f"[{ts}] HOLD  {self.coin} | Signal: {latest_signal} | Portfolio: ${portfolio:,.2f}")

    def _get_current_position(self) -> int:
        """Return +1 (long), -1 (short), or 0 (flat) for current position."""
        try:
            pos = self._trader.get_position(self.coin)
            if pos is None:
                return 0
            size = pos.get("size", 0)
            if size > 0:
                return 1
            if size < 0:
                return -1
        except Exception:
            pass
        return 0
