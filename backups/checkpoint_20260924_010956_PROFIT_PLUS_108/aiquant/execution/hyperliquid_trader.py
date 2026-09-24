"""
aiquant/execution/hyperliquid_trader.py
=========================================
Hyperliquid Paper Trading Execution Layer.

Hyperliquid is a high-performance on-chain perpetuals DEX.
Paper trading uses the Hyperliquid testnet — no real funds required.

Setup:
  1. Generate an Ethereum wallet (private key)
  2. Fund the testnet account at: https://app.hyperliquid-testnet.xyz/
  3. Set HYPERLIQUID_PRIVATE_KEY in your .env file

Features:
  - Market and limit order placement
  - Position management (open, close, reduce)
  - Real-time account state (balance, positions, PnL)
  - Order status tracking
  - Kelly-sized position entry
  - Full trade log with timestamps
"""

import os
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class HyperliquidPaperTrader:
    """
    Hyperliquid testnet paper trading execution layer.
    Connects to Hyperliquid's on-chain perpetuals DEX via their Python SDK.

    Usage:
        trader = HyperliquidPaperTrader()
        trader.connect()
        trader.place_market_order('BTC', 'long', size_usd=1000)
    """

    TESTNET_API = "https://api.hyperliquid-testnet.xyz"
    MAINNET_API = "https://api.hyperliquid.xyz"

    def __init__(
        self,
        private_key: Optional[str] = None,
        use_testnet: bool = True,
        log_dir: str = 'logs',
    ):
        self.private_key = private_key or os.getenv('HYPERLIQUID_PRIVATE_KEY')
        self.use_testnet = use_testnet
        self.base_url = self.TESTNET_API if use_testnet else self.MAINNET_API
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.exchange = None
        self.info = None
        self.address = None
        self.is_connected = False
        self.trade_log: List[Dict] = []

        logger.info(f"HyperliquidPaperTrader initialised ({'TESTNET' if use_testnet else 'MAINNET'})")

    def connect(self) -> bool:
        """
        Connect to Hyperliquid using the SDK.
        Requires a valid private key in .env or passed directly.
        """
        if not self.private_key:
            logger.error(
                "No private key found. Set HYPERLIQUID_PRIVATE_KEY in your .env file.\n"
                "To generate a testnet wallet:\n"
                "  from eth_account import Account\n"
                "  acct = Account.create()\n"
                "  print(acct.key.hex(), acct.address)\n"
                "Then fund it at https://app.hyperliquid-testnet.xyz/drip"
            )
            return False

        try:
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
            from eth_account import Account

            account = Account.from_key(self.private_key)
            self.address = account.address

            self.info = Info(self.base_url, skip_ws=True)
            self.exchange = Exchange(account, self.base_url)

            # Test connection
            state = self.info.user_state(self.address)
            balance = float(state.get('marginSummary', {}).get('accountValue', 0))
            logger.info(f"Connected to Hyperliquid | Address: {self.address} | Balance: ${balance:,.2f}")
            self.is_connected = True
            return True

        except ImportError:
            logger.error("Hyperliquid SDK not installed. Run: pip install hyperliquid")
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Account State
    # ------------------------------------------------------------------

    def get_account_state(self) -> Dict:
        """Fetch current account state: balance, positions, margin."""
        if not self.is_connected:
            return {}
        try:
            state = self.info.user_state(self.address)
            margin = state.get('marginSummary', {})
            positions = state.get('assetPositions', [])

            open_positions = []
            for pos in positions:
                p = pos.get('position', {})
                if float(p.get('szi', 0)) != 0:
                    open_positions.append({
                        'coin': p.get('coin'),
                        'size': float(p.get('szi', 0)),
                        'entry_price': float(p.get('entryPx', 0)),
                        'unrealised_pnl': float(p.get('unrealizedPnl', 0)),
                        'leverage': p.get('leverage', {}),
                    })

            return {
                'account_value': float(margin.get('accountValue', 0)),
                'total_margin_used': float(margin.get('totalMarginUsed', 0)),
                'total_ntl_pos': float(margin.get('totalNtlPos', 0)),
                'open_positions': open_positions,
            }
        except Exception as e:
            logger.error(f"Error fetching account state: {e}")
            return {}

    def get_position(self, coin: str = 'BTC') -> Optional[Dict]:
        """Get current position for a specific coin."""
        state = self.get_account_state()
        for pos in state.get('open_positions', []):
            if pos['coin'] == coin:
                return pos
        return None

    # ------------------------------------------------------------------
    # Order Placement
    # ------------------------------------------------------------------

    def place_market_order(
        self,
        coin: str,
        direction: str,   # 'long' or 'short'
        size_usd: float,
        reduce_only: bool = False,
    ) -> Dict:
        """
        Place a market order on Hyperliquid.
        size_usd: notional USD value of the position.
        """
        if not self.is_connected:
            logger.error("Not connected. Call connect() first.")
            return {'success': False, 'error': 'Not connected'}

        try:
            # Get current price
            mid_price = self._get_mid_price(coin)
            if mid_price is None:
                return {'success': False, 'error': 'Could not fetch price'}

            size_coins = round(size_usd / mid_price, 4)
            is_buy = direction.lower() == 'long'

            # Hyperliquid market order uses a very aggressive limit price
            slippage = 0.01  # 1% slippage tolerance
            limit_px = mid_price * (1 + slippage) if is_buy else mid_price * (1 - slippage)
            limit_px = round(limit_px, 1)

            result = self.exchange.order(
                coin,
                is_buy,
                size_coins,
                limit_px,
                {'limit': {'tif': 'Ioc'}},  # Immediate-or-cancel = market order
                reduce_only=reduce_only,
            )

            order_id = result.get('response', {}).get('data', {}).get('statuses', [{}])[0].get('resting', {}).get('oid')
            success = result.get('status') == 'ok'

            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'coin': coin,
                'direction': direction,
                'size_usd': size_usd,
                'size_coins': size_coins,
                'price': mid_price,
                'order_id': order_id,
                'success': success,
                'reduce_only': reduce_only,
                'raw_response': result,
            }
            self.trade_log.append(log_entry)
            self._save_trade_log()

            if success:
                logger.info(
                    f"{'BUY' if is_buy else 'SELL'} {size_coins:.4f} {coin} "
                    f"@ ~${mid_price:,.2f} | USD: ${size_usd:,.2f} | OID: {order_id}"
                )
            else:
                logger.warning(f"Order may have failed: {result}")

            return log_entry

        except Exception as e:
            logger.error(f"Order placement error: {e}")
            return {'success': False, 'error': str(e)}

    def close_position(self, coin: str = 'BTC') -> Dict:
        """Close the entire open position for a coin."""
        pos = self.get_position(coin)
        if pos is None or pos['size'] == 0:
            logger.info(f"No open position for {coin}")
            return {'success': True, 'message': 'No position to close'}

        direction = 'short' if pos['size'] > 0 else 'long'  # Opposite to close
        size_usd = abs(pos['size']) * self._get_mid_price(coin)
        return self.place_market_order(coin, direction, size_usd, reduce_only=True)

    def place_limit_order(
        self,
        coin: str,
        direction: str,
        size_usd: float,
        limit_price: float,
        time_in_force: str = 'Gtc',  # Good-till-cancelled
    ) -> Dict:
        """Place a limit order."""
        if not self.is_connected:
            return {'success': False, 'error': 'Not connected'}
        try:
            mid_price = self._get_mid_price(coin)
            size_coins = round(size_usd / mid_price, 4)
            is_buy = direction.lower() == 'long'

            result = self.exchange.order(
                coin,
                is_buy,
                size_coins,
                round(limit_price, 1),
                {'limit': {'tif': time_in_force}},
            )
            success = result.get('status') == 'ok'
            logger.info(f"Limit order {'placed' if success else 'FAILED'}: {direction} {size_coins} {coin} @ ${limit_price:,.2f}")
            return {'success': success, 'result': result}
        except Exception as e:
            logger.error(f"Limit order error: {e}")
            return {'success': False, 'error': str(e)}

    # ------------------------------------------------------------------
    # Live Signal Execution Loop
    # ------------------------------------------------------------------

    def run_live(
        self,
        signal_generator,  # Callable that returns (signal, size_usd)
        coin: str = 'BTC',
        poll_interval_sec: float = 60.0,
        max_iterations: Optional[int] = None,
    ):
        """
        Live execution loop: polls signal generator every interval and executes.
        signal_generator: callable() -> (signal: int, size_usd: float)
          signal = 1 (long), -1 (short), 0 (flat)
        """
        if not self.is_connected:
            logger.error("Not connected. Call connect() first.")
            return

        logger.info(f"Starting live execution loop for {coin} | Interval: {poll_interval_sec}s")
        iteration = 0
        current_position = 0  # Track local position state

        while True:
            try:
                signal, size_usd = signal_generator()
                account = self.get_account_state()
                portfolio_value = account.get('account_value', 0)

                logger.info(
                    f"[{datetime.utcnow().isoformat()}] "
                    f"Signal: {signal} | Portfolio: ${portfolio_value:,.2f}"
                )

                # Execute signal
                if signal == 1 and current_position <= 0:
                    if current_position < 0:
                        self.close_position(coin)
                    self.place_market_order(coin, 'long', size_usd)
                    current_position = 1

                elif signal == -1 and current_position >= 0:
                    if current_position > 0:
                        self.close_position(coin)
                    self.place_market_order(coin, 'short', size_usd)
                    current_position = -1

                elif signal == 0 and current_position != 0:
                    self.close_position(coin)
                    current_position = 0

                iteration += 1
                if max_iterations and iteration >= max_iterations:
                    logger.info("Max iterations reached. Stopping.")
                    break

                time.sleep(poll_interval_sec)

            except KeyboardInterrupt:
                logger.info("Execution loop stopped by user.")
                break
            except Exception as e:
                logger.error(f"Execution loop error: {e}")
                time.sleep(poll_interval_sec)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_mid_price(self, coin: str) -> Optional[float]:
        """Fetch current mid price for a coin."""
        try:
            mids = self.info.all_mids()
            price = mids.get(coin)
            return float(price) if price else None
        except Exception as e:
            logger.error(f"Error fetching price for {coin}: {e}")
            return None

    def _save_trade_log(self):
        """Persist trade log to disk."""
        path = self.log_dir / 'hyperliquid_trades.json'
        with open(path, 'w') as f:
            json.dump(self.trade_log, f, indent=2, default=str)

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        df = pd.DataFrame(self.trade_log)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')

    def print_account_summary(self):
        """Print a formatted account summary."""
        state = self.get_account_state()
        print("\n" + "=" * 50)
        print("  HYPERLIQUID ACCOUNT SUMMARY")
        print("=" * 50)
        print(f"  Account Value:    ${state.get('account_value', 0):>12,.2f}")
        print(f"  Margin Used:      ${state.get('total_margin_used', 0):>12,.2f}")
        print(f"  Notional Pos:     ${state.get('total_ntl_pos', 0):>12,.2f}")
        positions = state.get('open_positions', [])
        if positions:
            print(f"\n  Open Positions ({len(positions)}):")
            for p in positions:
                print(
                    f"    {p['coin']:<6} | Size: {p['size']:>10.4f} | "
                    f"Entry: ${p['entry_price']:>10,.2f} | "
                    f"uPnL: ${p['unrealised_pnl']:>10,.2f}"
                )
        else:
            print("  No open positions.")
        print("=" * 50 + "\n")
