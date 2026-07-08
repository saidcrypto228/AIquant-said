"""
AIQuant — HFT Statistical Arbitrage Framework
==============================================
AegisFintech | Apache 2.0 License

Usage:
    python3 run.py backtest                          # BTC, last 1825 days (5 years)
    python3 run.py backtest --pair ETHUSDT           # different pair
    python3 run.py backtest --days 30                # shorter window
    python3 run.py backtest --pair ETH --days 60     # pair shorthand works too
    python3 run.py live                              # live trading on Hyperliquid mainnet

Defaults (when no flags given):
    --pair   BTCUSDT
    --days   1825  (T-1825 days of 1m data = ~2.6M bars)

Data sources:
    Backtest : Binance Vision monthly CSVs (free, no API key) + Hyperliquid public API
    Live     : Hyperliquid public API for market data + mainnet for execution
               Requires HYPERLIQUID_PRIVATE_KEY in .env
"""

import sys
import os
import time
import json
import logging
import argparse
import warnings
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
sys.path.insert(0, str(ROOT))
RESULTS_DIR = ROOT / 'results'
DATA_DIR    = ROOT / 'data' / 'raw'
LOGS_DIR    = ROOT / 'logs'
CONFIG_DIR  = ROOT / 'config'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Run defaults (single source of truth — see aiquant/defaults.py) ───────────
from aiquant.defaults import (
    DEFAULT_PAIR, DEFAULT_DAYS, DEFAULT_CAPITAL, DEFAULT_LIVE_CAPITAL, DEFAULT_POLL,
)

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('aiquant')
logger.setLevel(logging.INFO)

# ── Numba JIT warm-up ────────────────────────────────────────────────────────
try:
    from aiquant.utils.fast_math import warmup as _nb_warmup
    _nb_warmup()
except Exception:
    pass

# ── Colour helpers + ML pipeline (shared, single source) ──────────────────────
from aiquant.utils.console import _c, BOLD, DIM, GREEN, RED, CYAN, YELLOW, WHITE
from aiquant.models.ensemble_pipeline import run_ml_backtest

# ── Banner ────────────────────────────────────────────────────────────────────
BANNER = f"""
{CYAN('╔══════════════════════════════════════════════════════════════╗')}
{CYAN('║')}  {BOLD(WHITE('AIQuant'))}  ·  HFT Statistical Arbitrage  ·  {DIM('AegisFintech')}      {CYAN('║')}
{CYAN('║')}  {DIM('Apache 2.0  ·  github.com/AegisFintech/AIQuant')}             {CYAN('║')}
{CYAN('╚══════════════════════════════════════════════════════════════╝')}
"""

def banner(mode: str, pair: str, days: int = None):
    print(BANNER)
    mode_str = {
        'backtest': '📊  BACKTEST  (ML Ensemble · Binance Vision + Hyperliquid)',
        'live':     '🔴  LIVE TRADING  (Hyperliquid Mainnet)',
    }.get(mode, mode.upper())
    print(f"  Mode  : {BOLD(mode_str)}")
    print(f"  Pair  : {BOLD(CYAN(pair))}")
    if days:
        print(f"  Window: {BOLD(str(days))} days  ({days * 1440:,} 1m bars)")
    print()


# ════════════════════════════════════════════════════════════════════════════
# PAIR NORMALISATION
# ════════════════════════════════════════════════════════════════════════════

VALID_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

def normalise_pair(raw: str) -> str:
    """Accept BTC, btc, BTCUSDT, btcusdt — always return e.g. BTCUSDT."""
    p = raw.upper().strip()
    if p in VALID_PAIRS:
        return p
    if not p.endswith('USDT'):
        p = p + 'USDT'
    if p not in VALID_PAIRS:
        print(f"  {YELLOW('⚠')}  Unknown pair '{raw}'. Defaulting to BTCUSDT.")
        return 'BTCUSDT'
    return p


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING — Binance Vision + Hyperliquid
# ════════════════════════════════════════════════════════════════════════════

def load_data(pair: str = 'BTCUSDT', days: int = 1825) -> pd.DataFrame:
    """
    Load 1m OHLCV data. Automatically prepares data if missing or stale.
    """
    from aiquant.data.preparer import ensure_data_prepared, OUT_PATH
    
    # Unified data preparation flow
    ensure_data_prepared(days=days, verbose=True)

    print(f"  {CYAN('↓')} Loading dataset from {OUT_PATH.name}...", end=' ', flush=True)
    df = pd.read_parquet(OUT_PATH)

    # Trim to requested window
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days + 1)
    df = df[df.index >= cutoff]

    print(f"{GREEN('✓')}")
    print(f"  {DIM(f'{len(df):,} bars  ·  {df.index[0].date()} → {df.index[-1].date()}')}")
    close_min = df['close'].min()
    close_max = df['close'].max()
    print(f"  {DIM(f'Price range: ${close_min:,.0f} → ${close_max:,.0f}')}")

    return df


# ════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all 183 features with verbose per-step timing."""
    from aiquant.features import build_full_feature_set
    return build_full_feature_set(df, verbose=True)




# ════════════════════════════════════════════════════════════════════════════
# LIVE TRADING — Hyperliquid Mainnet
# ════════════════════════════════════════════════════════════════════════════

def run_live_ml(pair: str = 'BTCUSDT', capital: float = 10_000, poll: float = 60.0):
    """
    Start ML-powered live trading on Hyperliquid mainnet.
    Loads the model bundle saved by 'python3 run.py backtest'.
    """
    bundle_path = ROOT / 'models' / 'ml_live_bundle.pkl'
    if not bundle_path.exists():
        print(f"\n  {RED('✗')}  Model bundle not found at models/ml_live_bundle.pkl")
        print(f"  {YELLOW('!')}  Run 'python3 run.py backtest' first to train and save the model.")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()
    pk = os.getenv('HYPERLIQUID_PRIVATE_KEY', '')
    if not pk or pk.startswith('your_'):
        print(f"\n  {RED('✗')}  HYPERLIQUID_PRIVATE_KEY not set in .env")
        print(f"  {DIM('  1. Open .env and add your Hyperliquid private key')}")
        print(f"  {DIM('  2. Generate a wallet: python3 -c \"from eth_account import Account; a=Account.create(); print(a.key.hex())\"')}")
        print(f"  {DIM('  3. Fund your account at https://app.hyperliquid.xyz')}")
        sys.exit(1)

    print(f"  {GREEN('✓')} Model bundle found: {bundle_path.name}")
    print(f"  {GREEN('✓')} Private key loaded")
    print(f"  {CYAN('▶')}  Starting ML live trading loop  (Ctrl+C to stop)")
    print(f"  {DIM('  Polling every ' + str(poll) + 's  ·  Max 25% position per trade')}")
    print(f"  {DIM('  Signals: XGBoost 40% + LightGBM 40% + LSTM 20%')}")
    print()

    from aiquant.execution.ml_live_trader import MLLiveTrader
    trader = MLLiveTrader(
        pair              = pair,
        initial_capital   = capital,
        kelly_fraction    = 0.5,
        poll_interval_sec = poll,
        feature_window    = 600,
        log_dir           = str(LOGS_DIR / 'ml_live'),
    )
    trader.start()


def run_live(pair: str = 'BTCUSDT', capital: float = 10_000, poll: float = 60.0):
    """Start live trading on Hyperliquid mainnet (rule-based fallback)."""
    from dotenv import load_dotenv
    load_dotenv()

    pk = os.getenv('HYPERLIQUID_PRIVATE_KEY', '')
    if not pk or pk.startswith('your_'):
        print(f"\n  {RED('✗')}  HYPERLIQUID_PRIVATE_KEY not set in .env")
        print(f"  {DIM('  1. Open .env and add your Hyperliquid private key')}")
        print(f"  {DIM('  2. Generate a wallet: python3 -c \"from eth_account import Account; a=Account.create(); print(a.key.hex())\"')}")
        print(f"  {DIM('  3. Fund your account at https://app.hyperliquid.xyz')}")
        sys.exit(1)

    coin = pair.replace('USDT', '')
    print(f"  {GREEN('✓')} Private key loaded")
    print(f"  {CYAN('▶')}  Starting live trading loop  (rule-based mode)  (Ctrl+C to stop)")
    print(f"  {DIM('  Polling every ' + str(poll) + 's  ·  Max 25% position per trade')}")
    print(f"  {YELLOW('!')}  Tip: run with --ml to use the trained ML ensemble instead")
    print()

    from aiquant.execution.live_trader import LiveTradingOrchestrator
    orchestrator = LiveTradingOrchestrator(
        pair              = pair,
        coin              = coin,
        initial_capital   = capital,
        kelly_fraction    = 0.5,
        poll_interval_sec = poll,
        log_dir           = str(LOGS_DIR / 'live_trading'),
    )
    orchestrator.start()


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        prog='python3 run.py',
        description='AIQuant — HFT Statistical Arbitrage Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run.py backtest                      # BTC, last 1825 days (5 years)
  python3 run.py backtest --pair ETH           # Ethereum, last 1825 days (5 years)
  python3 run.py backtest --days 30            # BTC, last 30 days
  python3 run.py backtest --pair SOL --days 60 # Solana, last 60 days
  python3 run.py backtest --fast               # Skip LSTM (faster, ~3x speedup)
  python3 run.py live --ml                     # ML live trading (after backtest)
  python3 run.py live --ml --pair ETH          # ETH with ML signals
  python3 run.py live --ml --poll 30           # ML mode, poll every 30s
  python3 run.py live                          # Rule-based live trading (fallback)
        """
    )
    sub = parser.add_subparsers(dest='mode', required=True)

    # ── backtest ──────────────────────────────────────────────────────────
    bt_p = sub.add_parser('backtest', help='Run ML ensemble backtest on Binance Vision data')
    bt_p.add_argument('--pair',    default=DEFAULT_PAIR, help=f'Trading pair (default: {DEFAULT_PAIR})')
    bt_p.add_argument('--days',    default=DEFAULT_DAYS, type=int, help=f'Days of history (default: {DEFAULT_DAYS} = 5 years)')
    bt_p.add_argument('--capital', default=DEFAULT_CAPITAL, type=float, help=f'Starting capital USD (default: {DEFAULT_CAPITAL})')
    bt_p.add_argument('--force',   action='store_true', help='Force re-download even if cache exists')
    bt_p.add_argument('--fast',    action='store_true', help='Skip LSTM training (faster, ~3x speedup)')

    # ── live ──────────────────────────────────────────────────────────────
    lv_p = sub.add_parser('live', help='Start live trading on Hyperliquid mainnet')
    lv_p.add_argument('--pair',    default=DEFAULT_PAIR, help=f'Trading pair (default: {DEFAULT_PAIR})')
    lv_p.add_argument('--capital', default=DEFAULT_LIVE_CAPITAL, type=float, help=f'Starting capital USD (default: {DEFAULT_LIVE_CAPITAL})')
    lv_p.add_argument('--poll',    default=DEFAULT_POLL, type=float, help=f'Poll interval in seconds (default: {DEFAULT_POLL:g})')
    lv_p.add_argument('--ml',      action='store_true',
                      help='Use trained ML ensemble (requires models/ml_live_bundle.pkl from backtest)')

    args = parser.parse_args()
    pair = normalise_pair(args.pair)

    if args.mode == 'backtest':
        banner('backtest', pair, args.days)
        t0 = time.time()

        # Step 1: Load data
        print(f"  {CYAN('━'*54)}")
        print(f"  Step 1 / 3  ·  Loading Data")
        print(f"  {CYAN('━'*54)}")
        df = load_data(pair=pair, days=args.days)

        # Step 2: Build features
        print(f"\n  {CYAN('━'*54)}")
        print(f"  Step 2 / 3  ·  Feature Engineering  (183 features)")
        print(f"  {CYAN('━'*54)}")
        df_feat = build_features(df)
        df_feat = df_feat.dropna()
        print(f"  {GREEN('✓')} {len(df_feat):,} bars × {df_feat.shape[1]} features after dropna")

        # Step 3: ML ensemble backtest
        print(f"\n  {CYAN('━'*54)}")
        print(f"  Step 3 / 3  ·  ML Ensemble Backtest  (XGB+LGB+LSTM)")
        print(f"  {CYAN('━'*54)}")
        results = run_ml_backtest(
            df_feat, pair=pair, capital=args.capital,
            fast=args.fast, days=args.days
        )

        elapsed = time.time() - t0
        print(f"\n  {DIM(f'Total time: {elapsed:.1f}s  ({elapsed/60:.1f} min)')}")
        print(f"  {DIM('Best params saved → config/ml_best_params.json')}")
        print(f"  {DIM('Model bundle saved → models/ml_live_bundle.pkl')}")
        print(f"  {DIM('Chart saved → results/backtest_results.png')}")
        print(f"\n  {GREEN('✓')} Ready to trade! Run: {BOLD('python3 run.py live --ml')}")
        print(f"  {DIM('  (set HYPERLIQUID_PRIVATE_KEY in .env first)')}")
        print()

    elif args.mode == 'live':
        banner('live', pair)
        if args.ml:
            run_live_ml(pair=pair, capital=args.capital, poll=args.poll)
        else:
            run_live(pair=pair, capital=args.capital, poll=args.poll)


if __name__ == '__main__':
    main()
