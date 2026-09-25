# AIQuant — HFT Statistical Arbitrage Framework

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AegisFintech/AIQuant/blob/main/AIQuant_Colab.ipynb)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)

**AegisFintech** · Professional-grade quantitative trading system for BTCUSD on Hyperliquid.

---

## Default: 5-Year Backtest (1825 days · Jan 2021 – Jun 2026 · ~2.63M bars)

The system defaults to **5 years** of BTCUSDT 1-minute data from Binance Vision.
The table below shows validated results on the most recent 365-day window:

| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **+0.625** |
| Total Return | +6.26% |
| Max Drawdown | -7.87% |
| Calmar Ratio | 0.795 |
| Win Rate | 63.5% |
| Total Trades | 19,302 |
| Profit Factor | 1.024x |
| Dataset | 530,628 bars · BTC $60k → $126k |

> **Note:** The 365-day window includes the Jun–Nov 2025 bear market (BTC -52%). The ML ensemble
> achieves Sharpe **3.39** on the bull-run period (Mar–Jun 2026) when trained on that window alone.
> With 5 years of data the model learns multiple full market cycles (2021 bull, 2022 bear, 2023–2024 recovery, 2025–2026 bull).

---

## Quick Start

### Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AegisFintech/AIQuant/blob/main/AIQuant_Colab.ipynb)

1. Click the badge above
2. Set runtime to GPU (`Runtime → Change runtime type → T4 GPU`)
3. Run all cells in order — **Step 3** lets you change `DAYS` (default `1825`)

### Local Installation

```bash
git clone https://github.com/AegisFintech/AIQuant.git
cd AIQuant
pip install -r requirements.txt
```

### Data Setup

Place yearly zip files (provided or downloaded from [Binance Vision](https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/)) into the `data/` folder, then run:

```bash
# Auto-detects local zips in data/ and extracts them.
# If no zips are found, downloads from Binance Vision automatically.
# Also fetches the last 7 days from Hyperliquid to bridge the gap to today.
python3 scripts/prepare_data.py

# Custom window (default is 1825 = 5 years)
python3 scripts/prepare_data.py --days 365
```

Accepted zip filenames: `BTCUSDT_1m_2021.zip`, `21.zip`, or any `*.zip` containing `BTCUSDT-1m-YYYY-MM.csv` files.

### CLI Usage

```bash
# Full 5-year ML ensemble backtest (default)
python3 run.py backtest

# Faster mode (skip LSTM, ~3x speedup)
python3 run.py backtest --fast

# Custom window
python3 run.py backtest --days 365

# Live trading with ML model (run backtest first to save model)
python3 run.py live --ml

# Live trading with rule-based strategy
python3 run.py live
```

---

## Architecture

```
AIQuant/
├── run.py                          # CLI entry point (backtest + live)
├── qvex/
│   ├── data/
│   │   └── fetcher.py              # Binance Vision + Hyperliquid data
│   ├── features/
│   │   ├── technical.py            # 92 technical indicators (fast NumPy)
│   │   ├── microstructure.py       # 55 microstructure features
│   │   ├── statarb.py              # 19 StatArb / regime features
│   │   └── gpu_features.py         # CuPy GPU-accelerated feature engineering
│   ├── models/
│   │   └── ensemble_pipeline.py    # ML pipeline: XGBoost + LightGBM + LSTM, walk-forward CV
│   ├── execution/
│   │   ├── hyperliquid_trader.py   # Hyperliquid mainnet execution
│   │   ├── ml_live_trader.py       # ML live trading (loads saved model bundle)
│   │   └── live_trader.py          # Rule-based live trading orchestrator
│   ├── risk/
│   │   └── manager.py              # Kelly Criterion + drawdown limits
│   └── utils/
│       ├── console.py              # Shared ANSI colour helpers
│       └── fast_math.py            # Numba JIT: Hurst, ADF, Kalman, OU
├── scripts/
│   ├── prepare_data.py             # Build Binance Vision dataset
│   ├── install_hooks.sh            # Install pre-commit hook (keeps notebook in sync)
│   └── build_colab.py              # Regenerate AIQuant_Colab.ipynb
├── models/
│   └── ml_live_bundle.pkl          # Saved ML model bundle (after backtest)
├── config/
│   ├── settings.yaml               # System configuration
│   └── ml_best_params.json         # Saved ML best parameters
└── data/raw/                       # OHLCV data (gitignored)
```

---

## Keeping Local & Colab in Sync

There is **one** codebase, not two. The Colab notebook contains no strategy logic — it clones this
repo and shells out to the same `run.py` CLI the local path uses. Points to remember:

- **Colab runs GitHub, not your laptop.** Step 1 of the notebook does `git clone ...` of the `main`
  branch. **Commit and push** before Colab picks up local changes.
- **Run defaults live in one place:** [`qvex/defaults.py`](qvex/defaults.py) (`PAIR`, `DAYS`,
  `CAPITAL`, ...). `run.py` and the notebook's Step 3 both read from it.
- **The notebook is generated**, not hand-edited: `scripts/build_colab.py` produces
  `AIQuant_Colab.ipynb`. Run the installer once so a pre-commit hook regenerates + stages it on every
  commit (a stale notebook then can't be committed):

  ```bash
  sh scripts/install_hooks.sh   # sets git core.hooksPath = scripts/hooks
  ```

  Edit `scripts/build_colab.py` (or `qvex/defaults.py`) to change the notebook — never edit the
  `.ipynb` by hand.

---

## ML Ensemble Pipeline

The backtest uses a **walk-forward cross-validation** pipeline with no lookahead bias:

1. **Labels** — 15-bar forward return, threshold 0.08% net of fees
2. **Feature selection** — Top 60 features by mutual information (from 171 total)
3. **Walk-forward folds** — Dynamic sizing: targets ~50 folds regardless of dataset length
   - 365 days → 60d train / 6d step / ~50 folds
   - 1825 days (5y) → 90d train / 30d step / ~57 folds
4. **XGBoost** — 200 estimators, depth 5, class-balanced weights, CUDA GPU
5. **LightGBM** — 200 estimators, 31 leaves, class-balanced weights, GPU
6. **LSTM + Attention** — 30-bar sequences, 2-layer LSTM, 64 hidden units, PyTorch CUDA
7. **Ensemble** — XGB 40% + LGB 40% + LSTM 20%
8. **Threshold search** — Grid search over long/short confidence thresholds
9. **Model saving** — Best model bundle saved to `models/ml_live_bundle.pkl` for live trading

---

## Data Sources

| Source | Coverage | Auth Required |
|--------|----------|---------------|
| [Binance Vision](https://data.binance.vision/) | Monthly CSVs, Jan 2017–present | None |
| [Hyperliquid](https://app.hyperliquid.xyz/) | Real-time candles (last 7 days) | None (read) |

**Data scaling by `DAYS` setting:**

| DAYS | Files | Approx size | Approx bars |
|------|-------|-------------|-------------|
| 365 (1 year) | 12 | ~40 MB | 530k |
| 730 (2 years) | 24 | ~80 MB | 1.05M |
| 1095 (3 years) | 36 | ~120 MB | 1.58M |
| **1825 (5 years, default)** | **60** | **~200 MB** | **2.63M** |

---

## Live Trading

### ML Live Trading (Recommended — uses trained model)

```bash
# Step 1: Train and save the model bundle
python3 run.py backtest

# Step 2: Start ML live trading
python3 run.py live --ml
```

### Rule-Based Live Trading

```bash
# Requires Hyperliquid mainnet account
echo "HYPERLIQUID_PRIVATE_KEY=0x..." >> .env
python3 run.py live
```

---

## Configuration

Edit `config/settings.yaml` or set environment variables in `.env`:

```yaml
pair: BTCUSDT
interval: 1m
kelly_fraction: 0.5          # Half-Kelly position sizing
max_position_pct: 0.25       # Max 25% of capital per trade
max_drawdown_pct: 0.15       # Stop trading at 15% drawdown
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves
substantial risk of loss. Past performance does not guarantee future results.