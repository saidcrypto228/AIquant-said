"""
scripts/build_colab.py
=======================
Rebuilds AIQuant_Colab.ipynb with the simplified Zero-Configuration pipeline.
"""

import json
import sys
from pathlib import Path

ROOT    = Path(__file__).parent.parent
NB_PATH = ROOT / 'AIQuant_Colab.ipynb'

# Pull run defaults from the single source of truth so the notebook's Step 3
# config cell never drifts from run.py — see aiquant/defaults.py.
sys.path.insert(0, str(ROOT))
from aiquant.defaults import (
    DEFAULT_PAIR, DEFAULT_DAYS, DEFAULT_CAPITAL, DEFAULT_FAST,
)

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.split('\n')}

def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in source.split('\n')],
    }

cells = []

# ── Header ────────────────────────────────────────────────────────────────────
cells.append(md("""# AIQuant — HFT Statistical Arbitrage
**AegisFintech · Apache 2.0 · [github.com/AegisFintech/AIQuant](https://github.com/AegisFintech/AIQuant)**

> **GPU Runtime Recommended** — Go to `Runtime → Change runtime type → GPU (T4 or A100)` for faster ML training.

| Step | Description |
|------|-------------|
| 1 | Clone repository |
| 2 | Install dependencies + verify GPU |
| 3 | Configuration (Pair, Days, Capital) |
| 4 | **Unified Backtest** (Auto-Data Download + Feature Engineering + ML Training) |
| 5 | View results chart |
| 6 | Live Trading (Hyperliquid mainnet) |
| 7 | Download results |

**Zero-Configuration Flow:** The backtest step now automatically handles data downloading from Binance Vision, local zip extraction, and gap-bridging via Hyperliquid. No manual data preparation required."""))

# ── Step 1: Clone ─────────────────────────────────────────────────────────────
cells.append(md("## Step 1 — Clone Repository"))
cells.append(code("""import os, sys, shutil
from pathlib import Path

os.chdir('/content')
!rm -rf AIQuant
!git clone https://github.com/AegisFintech/AIQuant.git
os.chdir('/content/AIQuant')

# Purge pycache
for p in Path('/content/AIQuant').rglob('__pycache__'):
    shutil.rmtree(p, ignore_errors=True)

!git log --oneline -3
print('\\n✓ Repository ready')"""))

# ── Step 2: Install + GPU check ───────────────────────────────────────────────
cells.append(md("## Step 2 — Install Dependencies & Verify GPU"))
cells.append(code("""!pip install -q -r requirements.txt
!pip install -q cupy-cuda12x xgboost lightgbm torch ta numba pyarrow statsmodels scikit-learn python-dotenv python-dateutil jupyter-server==2.18.2

print('\\n' + '='*60)
print('  GPU & Library Verification')
print('='*60)

try:
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f'  ✓ GPU  : {name}')
    else:
        print('  ✗ GPU  : NOT available — training will be slow!')
except Exception:
    pass

import xgboost as xgb
import lightgbm as lgb
print(f'  ✓ XGBoost   : {xgb.__version__}')
print(f'  ✓ LightGBM  : {lgb.__version__}')
print('='*60)"""))

# ── Step 3: Config ────────────────────────────────────────────────────────────
cells.append(md("## Step 3 — Configuration"))
# Settings values below are interpolated from aiquant/defaults.py (single source of truth).
step3_settings = f"""import os, sys
sys.path.insert(0, '/content/AIQuant')
os.chdir('/content/AIQuant')

# ── Settings ──────────────────────────────────────────────────────────────────
PAIR                    = '{DEFAULT_PAIR}'   # Trading pair
DAYS                    = {DEFAULT_DAYS}        # Days of backtest data (1825 = 5 years)
INITIAL_CAPITAL         = {DEFAULT_CAPITAL}     # Starting capital (USD)
HYPERLIQUID_PRIVATE_KEY = ''          # Required for Step 6 (Live Trading)
FAST_MODE               = {DEFAULT_FAST}       # True = skip LSTM (~3x faster)
"""
step3_env = """
# ── Write .env ────────────────────────────────────────────────────────────────
with open('.env', 'w') as f:
    f.write(f"HYPERLIQUID_PRIVATE_KEY={HYPERLIQUID_PRIVATE_KEY}\\n")
    f.write("HYPERLIQUID_TESTNET=false\\n")
    f.write(f"TRADING_PAIR={PAIR}\\n")
    f.write(f"INITIAL_CAPITAL={INITIAL_CAPITAL}\\n")

print(f'✓ Config set: {PAIR} | {DAYS} days | ${INITIAL_CAPITAL:,} capital')"""
cells.append(code(step3_settings + step3_env))

# ── Step 4: Unified Backtest ──────────────────────────────────────────────────
cells.append(md("""## Step 4 — Unified Backtest
This single step now handles:
1. **Auto-Data Preparation**: Detects local zips, downloads from Binance Vision, and bridges via Hyperliquid.
2. **Feature Engineering**: 183 technical, microstructure, and statarb features.
3. **ML Training**: XGBoost + LightGBM + LSTM walk-forward ensemble.
"""))
cells.append(code("""import subprocess, sys, time, os

# If you have local zip files (e.g. 21.zip, 22.zip), 
# upload them to the 'data/' folder using the Files panel on the left 
# BEFORE running this cell.

cmd = [sys.executable, '-u', 'run.py', 'backtest',
       '--pair', PAIR, '--days', str(DAYS),
       '--capital', str(INITIAL_CAPITAL)]
if FAST_MODE: cmd.append('--fast')

print(f'Running Unified Pipeline: {" ".join(cmd)}\\n')

process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                           text=True, bufsize=1, universal_newlines=True,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})

for line in process.stdout:
    print(line, end='', flush=True)

process.wait()
if process.returncode == 0:
    print(f'\\n✓ Pipeline completed successfully')
else:
    print(f'\\n✗ Pipeline failed with exit code {process.returncode}')"""))

# ── Step 5: Results ───────────────────────────────────────────────────────────
cells.append(md("## Step 5 — View Results Chart"))
cells.append(code("""from IPython.display import Image, display
res_path = '/content/AIQuant/results/backtest_results.png'
if os.path.exists(res_path):
    display(Image(filename=res_path))
else:
    print('✗ Backtest results chart not found. Ensure Step 4 completed successfully.')"""))

# ── Step 6: Live Trading ──────────────────────────────────────────────────────
cells.append(md("## Step 6 — Live Trading (Hyperliquid)"))
cells.append(code("""if not HYPERLIQUID_PRIVATE_KEY:
    print('✗ HYPERLIQUID_PRIVATE_KEY is missing in Step 3.')
else:
    cmd = [sys.executable, '-u', 'run.py', 'live', '--ml', '--pair', PAIR]
    print(f'Starting ML Live Trader: {" ".join(cmd)}\\n')
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                               text=True, bufsize=1, universal_newlines=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
    except KeyboardInterrupt:
        process.terminate()
        print('\\nStopping Live Trader...')"""))

# ── Step 7: Download ──────────────────────────────────────────────────────────
cells.append(md("## Step 7 — Download Results"))
cells.append(code("""from google.colab import files
import os

files_to_download = [
    '/content/AIQuant/results/backtest_results.png',
    '/content/AIQuant/config/ml_best_params.json',
    '/content/AIQuant/models/ml_live_bundle.pkl',
    '/content/AIQuant/logs/trade_log.json'
]

for f in files_to_download:
    if os.path.exists(f):
        files.download(f)
    else:
        print(f'Skipping {f} (not found)')"""))

# ── Finalize ──────────────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.10.12"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

with open(NB_PATH, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f'✓ Regenerated {NB_PATH.name}')
