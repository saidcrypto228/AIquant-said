"""
aiquant/defaults.py
===================
Single source of truth for run-time defaults shared across the local CLI
(`run.py`) and the generated Colab notebook (`scripts/build_colab.py`).

Change a value here and both the `run.py` argparse defaults and the notebook's
Step 3 configuration cell update from it — there is no second copy to keep in sync.
"""

DEFAULT_PAIR         = 'BTCUSDT'   # Trading pair
DEFAULT_DAYS         = 1825        # Days of backtest history (1825 = 5 years of 1m bars)
DEFAULT_CAPITAL      = 100_000     # Backtest starting capital (USD)
DEFAULT_LIVE_CAPITAL = 10_000      # Live starting capital (USD)
DEFAULT_POLL         = 60.0        # Live poll interval (seconds)
DEFAULT_FAST         = False       # Skip LSTM training (~3x faster) when True
