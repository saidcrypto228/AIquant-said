"""
aiquant/execution/ml_live_trader.py
=====================================
ML-powered live trading loop for AIQuant.

Workflow every `poll_interval_sec` seconds:
  1. Fetch latest 1m candles from Hyperliquid public API (no auth needed)
  2. Build all 183 features on the rolling window
  3. Scale features with the saved RobustScaler
  4. Run XGBoost + LightGBM (+ LSTM if available) → ensemble confidence score
  5. Apply saved thresholds → signal (+1 long / -1 short / 0 flat)
  6. Size position with half-Kelly criterion (max 25% of portfolio per trade)
  7. Execute on Hyperliquid mainnet via HyperliquidPaperTrader

Model bundle (models/ml_live_bundle.pkl) is created by:
    python3 run.py backtest

Usage:
    python3 run.py live --ml                        # ML mode (default after backtest)
    python3 run.py live --ml --pair ETH             # ETH with ML signals
    python3 run.py live --ml --poll 30              # poll every 30s
    python3 run.py live                             # old rule-based mode (fallback)
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROOT       = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / 'models'
LOGS_DIR   = ROOT / 'logs'


# ════════════════════════════════════════════════════════════════════════════
# MODEL BUNDLE LOADER
# ════════════════════════════════════════════════════════════════════════════

class MLModelBundle:
    """
    Loads and holds the trained model bundle saved by run.py backtest.

    Bundle keys:
        xgb           : trained XGBClassifier (last fold)
        lgb           : trained LGBMClassifier (last fold)
        scaler        : fitted RobustScaler
        top_features  : list of 60 feature column names
        lstm_features : list of 20 feature column names for LSTM
        lstm_seq_len  : int (30)
        long_thresh   : float (e.g. 0.08)
        short_thresh  : float (e.g. -0.18)
        lstm_state    : OrderedDict or None (PyTorch state_dict)
        lstm_feat_mean: np.ndarray or None
        lstm_feat_std : np.ndarray or None
        trained_at    : ISO timestamp string
        pair          : str (e.g. 'BTCUSDT')
    """

    def __init__(self, bundle_path: Optional[str] = None):
        self.path = Path(bundle_path) if bundle_path else MODELS_DIR / 'ml_live_bundle.pkl'
        self.bundle = None
        self._lstm_model = None
        self._torch_device = None

    def load(self) -> bool:
        """Load the bundle from disk. Returns True on success."""
        if not self.path.exists():
            logger.error(
                f"Model bundle not found at {self.path}. "
                "Run 'python3 run.py backtest' first to train and save the model."
            )
            return False
        try:
            import joblib
            self.bundle = joblib.load(self.path)
            trained_at = self.bundle.get('trained_at', 'unknown')
            pair       = self.bundle.get('pair', 'unknown')
            has_lstm   = self.bundle.get('lstm_state') is not None
            logger.info(
                f"Model bundle loaded | pair={pair} | trained={trained_at} | "
                f"LSTM={'yes' if has_lstm else 'no'} | "
                f"long_thresh={self.bundle['long_thresh']} | "
                f"short_thresh={self.bundle['short_thresh']}"
            )
            self._init_lstm()
            return True
        except Exception as e:
            logger.error(f"Failed to load model bundle: {e}")
            return False

    def _init_lstm(self):
        """Reconstruct the LSTM model from saved state dict."""
        if self.bundle is None or self.bundle.get('lstm_state') is None:
            return
        try:
            import torch
            import torch.nn as nn

            class LSTMAttn(nn.Module):
                def __init__(self, input_size, hidden=64, n_layers=2, n_classes=3):
                    super().__init__()
                    self.lstm = nn.LSTM(input_size, hidden, n_layers,
                                        batch_first=True, dropout=0.2)
                    self.attn = nn.Linear(hidden, 1)
                    self.fc   = nn.Sequential(
                        nn.Linear(hidden, 32), nn.ReLU(),
                        nn.Dropout(0.2), nn.Linear(32, n_classes)
                    )
                def forward(self, x):
                    out, _ = self.lstm(x)
                    attn_w = torch.softmax(self.attn(out), dim=1)
                    ctx    = (out * attn_w).sum(dim=1)
                    return self.fc(ctx)

            n_feats = len(self.bundle['lstm_features'])
            self._torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._lstm_model   = LSTMAttn(n_feats).to(self._torch_device)
            self._lstm_model.load_state_dict(self.bundle['lstm_state'])
            self._lstm_model.eval()
            logger.info(f"LSTM model loaded onto {self._torch_device}")
        except Exception as e:
            logger.warning(f"LSTM init failed (will use XGB+LGB only): {e}")
            self._lstm_model = None

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, df_feat: pd.DataFrame) -> tuple:
        """
        Generate signal and confidence score from the latest row of features.

        Parameters
        ----------
        df_feat : DataFrame with all features built (output of build_full_feature_set)

        Returns
        -------
        (signal: int, score: float, breakdown: dict)
          signal = +1 (long), -1 (short), 0 (flat)
          score  = ensemble confidence score
          breakdown = {'xgb': float, 'lgb': float, 'lstm': float or None}
        """
        if self.bundle is None:
            raise RuntimeError("Bundle not loaded. Call load() first.")

        top_features  = self.bundle['top_features']
        lstm_features = self.bundle['lstm_features']
        seq_len       = self.bundle['lstm_seq_len']
        long_thresh   = self.bundle['long_thresh']
        short_thresh  = self.bundle['short_thresh']
        scaler        = self.bundle['scaler']
        xgb_model     = self.bundle['xgb']
        lgb_model     = self.bundle['lgb']

        # Ensure all required features are present
        missing = [f for f in top_features if f not in df_feat.columns]
        if missing:
            raise ValueError(f"Missing {len(missing)} features in df_feat: {missing[:5]}...")

        # Use the most recent row for XGB + LGB
        X_raw = df_feat[top_features].to_numpy(np.float64)
        X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = scaler.transform(X_raw[-1:])   # shape (1, n_features)

        xgb_proba = xgb_model.predict_proba(X_scaled)[0]   # [short, flat, long]
        lgb_proba = lgb_model.predict_proba(X_scaled)[0]

        xgb_score = float(xgb_proba[2] - xgb_proba[0])
        lgb_score = float(lgb_proba[2] - lgb_proba[0])

        # LSTM: use the last seq_len rows
        lstm_score = None
        if self._lstm_model is not None and len(df_feat) >= seq_len:
            try:
                import torch
                X_lstm_raw = df_feat[lstm_features].to_numpy(np.float64)
                X_lstm_raw = np.nan_to_num(X_lstm_raw, nan=0.0, posinf=0.0, neginf=0.0)
                seq        = X_lstm_raw[-seq_len:]   # shape (seq_len, n_lstm_feats)
                X_t        = torch.tensor(seq[np.newaxis], dtype=torch.float32).to(self._torch_device)

                # Normalise using saved train statistics
                fm = torch.tensor(self.bundle['lstm_feat_mean'], dtype=torch.float32).to(self._torch_device)
                fs = torch.tensor(self.bundle['lstm_feat_std'],  dtype=torch.float32).to(self._torch_device)
                X_t = (X_t - fm) / fs

                with torch.no_grad():
                    proba = torch.softmax(self._lstm_model(X_t), dim=1).cpu().numpy()[0]
                lstm_score = float(proba[2] - proba[0])
            except Exception as e:
                logger.warning(f"LSTM inference failed: {e}")

        # Ensemble
        if lstm_score is not None:
            ens_score = 0.40 * xgb_score + 0.40 * lgb_score + 0.20 * lstm_score
        else:
            ens_score = 0.50 * xgb_score + 0.50 * lgb_score

        # Threshold → signal
        if ens_score > long_thresh:
            signal = 1
        elif ens_score < short_thresh:
            signal = -1
        else:
            signal = 0

        breakdown = {'xgb': round(xgb_score, 4), 'lgb': round(lgb_score, 4),
                     'lstm': round(lstm_score, 4) if lstm_score is not None else None}
        return signal, round(ens_score, 4), breakdown

    @property
    def is_loaded(self) -> bool:
        return self.bundle is not None

    @property
    def trained_at(self) -> str:
        return self.bundle.get('trained_at', 'unknown') if self.bundle else 'not loaded'

    @property
    def long_thresh(self) -> float:
        return self.bundle['long_thresh'] if self.bundle else 0.08

    @property
    def short_thresh(self) -> float:
        return self.bundle['short_thresh'] if self.bundle else -0.18


# ════════════════════════════════════════════════════════════════════════════
# ML LIVE TRADING ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

class MLLiveTrader:
    """
    Live trading loop powered by the ML ensemble model bundle.

    Parameters
    ----------
    pair              : Trading pair, e.g. 'BTCUSDT'
    initial_capital   : Starting capital in USD
    kelly_fraction    : Kelly fraction for position sizing (default 0.5)
    poll_interval_sec : Seconds between each trading loop tick
    feature_window    : Number of recent bars to fetch for feature computation
    bundle_path       : Path to ml_live_bundle.pkl (default: models/ml_live_bundle.pkl)
    log_dir           : Directory for trade logs
    """

    def __init__(
        self,
        pair:              str   = 'BTCUSDT',
        initial_capital:   float = 10_000.0,
        kelly_fraction:    float = 0.5,
        poll_interval_sec: float = 60.0,
        feature_window:    int   = 600,
        bundle_path:       Optional[str] = None,
        log_dir:           str   = 'logs/ml_live',
    ):
        self.pair              = pair.upper()
        self.coin              = pair.upper().replace('USDT', '')
        self.initial_capital   = initial_capital
        self.kelly_fraction    = kelly_fraction
        self.poll_interval_sec = poll_interval_sec
        self.feature_window    = feature_window
        self.log_dir           = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._model  = MLModelBundle(bundle_path)
        self._trader = None
        self._current_position = 0   # +1 long, -1 short, 0 flat
        self._tick_count = 0
        self._trade_log  = []

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, max_iterations: Optional[int] = None):
        """
        Start the ML live trading loop.
        Blocks until interrupted (Ctrl+C) or max_iterations reached.
        """
        self._init()
        logger.info(
            f"ML Live Trading started | {self.pair} | "
            f"Capital: ${self.initial_capital:,.2f} | "
            f"Poll: {self.poll_interval_sec}s | "
            f"Model trained: {self._model.trained_at}"
        )
        print(f"\n  Model trained at : {self._model.trained_at}")
        print(f"  Long threshold   : {self._model.long_thresh}")
        print(f"  Short threshold  : {self._model.short_thresh}")
        print(f"  LSTM available   : {'yes' if self._model._lstm_model is not None else 'no'}")
        print(f"\n  Press Ctrl+C to stop.\n")

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
                print("\n  Stopped by user.")
                self._save_trade_log()
                break
            except Exception as e:
                logger.error(f"Tick error: {e}", exc_info=True)
                time.sleep(self.poll_interval_sec)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init(self):
        """Load model bundle and connect to Hyperliquid."""
        # Load model
        if not self._model.load():
            raise RuntimeError(
                "Could not load ML model bundle. "
                "Run 'python3 run.py backtest' first."
            )

        # Connect to Hyperliquid
        from dotenv import load_dotenv
        load_dotenv()
        pk = os.getenv('HYPERLIQUID_PRIVATE_KEY', '')
        if not pk or pk.startswith('your_'):
            raise RuntimeError(
                "HYPERLIQUID_PRIVATE_KEY not set in .env\n"
                "  1. Open .env and add your Hyperliquid private key\n"
                "  2. Fund your account at https://app.hyperliquid.xyz"
            )

        from .hyperliquid_trader import HyperliquidPaperTrader
        self._trader = HyperliquidPaperTrader(
            use_testnet=False,
            log_dir=str(self.log_dir),
        )
        if not self._trader.connect():
            raise RuntimeError(
                "Could not connect to Hyperliquid. "
                "Check your HYPERLIQUID_PRIVATE_KEY."
            )
        logger.info("Connected to Hyperliquid mainnet.")

    def _tick(self):
        """Single trading loop iteration: fetch → features → signal → execute."""
        from ..data.fetcher import fetch_hyperliquid_candles
        from ..features import build_full_feature_set

        self._tick_count += 1
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')

        # 1. Fetch latest bars
        df_raw = fetch_hyperliquid_candles(
            pair=self.pair,
            n_bars=self.feature_window,
            interval='1m',
        )
        if len(df_raw) < 100:
            logger.warning(f"[{ts}] Only {len(df_raw)} bars fetched — skipping tick.")
            return

        # 2. Build features
        df_feat = build_full_feature_set(df_raw, verbose=False)
        df_feat = df_feat.dropna()
        if len(df_feat) < 50:
            logger.warning(f"[{ts}] Too few valid rows after feature build — skipping tick.")
            return

        # 3. Generate ML signal
        signal, score, breakdown = self._model.predict(df_feat)

        # 4. Compute Kelly-sized position
        account   = self._trader.get_account_state()
        portfolio = account.get('account_value', self.initial_capital)

        close_arr = df_feat['close'].to_numpy(dtype=np.float64)
        log_ret   = np.diff(np.log(close_arr[-21:]))
        vol_20    = float(np.std(log_ret) * np.sqrt(1440)) if len(log_ret) > 1 else 0.02
        size_usd  = portfolio * self.kelly_fraction * max(0.05, 1.0 - vol_20 * 5)
        size_usd  = min(size_usd, portfolio * 0.25)   # hard cap: max 25% per trade

        current_price = float(close_arr[-1])

        # 5. Log tick
        lstm_str = f"lstm={breakdown['lstm']}" if breakdown['lstm'] is not None else 'lstm=n/a'
        logger.info(
            f"[{ts}] tick={self._tick_count} | "
            f"price=${current_price:,.2f} | "
            f"score={score:+.4f} (xgb={breakdown['xgb']:+.4f} lgb={breakdown['lgb']:+.4f} {lstm_str}) | "
            f"signal={'LONG' if signal==1 else 'SHORT' if signal==-1 else 'FLAT'} | "
            f"pos={self._current_position} | "
            f"portfolio=${portfolio:,.2f}"
        )
        print(
            f"  [{ts}]  price=${current_price:,.2f}  "
            f"score={score:+.4f}  "
            f"signal={'LONG ' if signal==1 else 'SHORT' if signal==-1 else 'FLAT '}  "
            f"portfolio=${portfolio:,.2f}",
            flush=True
        )

        # 6. Execute
        if signal == 1 and self._current_position <= 0:
            if self._current_position < 0:
                self._trader.close_position(self.coin)
                logger.info(f"[{ts}] Closed SHORT before going LONG")
            self._trader.place_market_order(self.coin, 'long', size_usd)
            self._current_position = 1
            self._log_trade('LONG', current_price, size_usd, score, portfolio)
            logger.info(f"[{ts}] LONG  {self.coin} | ${size_usd:,.0f}")

        elif signal == -1 and self._current_position >= 0:
            if self._current_position > 0:
                self._trader.close_position(self.coin)
                logger.info(f"[{ts}] Closed LONG before going SHORT")
            self._trader.place_market_order(self.coin, 'short', size_usd)
            self._current_position = -1
            self._log_trade('SHORT', current_price, size_usd, score, portfolio)
            logger.info(f"[{ts}] SHORT {self.coin} | ${size_usd:,.0f}")

        elif signal == 0 and self._current_position != 0:
            self._trader.close_position(self.coin)
            self._current_position = 0
            self._log_trade('FLAT', current_price, 0.0, score, portfolio)
            logger.info(f"[{ts}] FLAT  {self.coin} | closed position")

        # else: signal unchanged, hold current position

    def _log_trade(self, action: str, price: float, size_usd: float,
                   score: float, portfolio: float):
        self._trade_log.append({
            'timestamp': datetime.utcnow().isoformat(),
            'action':    action,
            'price':     price,
            'size_usd':  round(size_usd, 2),
            'score':     score,
            'portfolio': round(portfolio, 2),
        })

    def _save_trade_log(self):
        if not self._trade_log:
            return
        try:
            import json
            log_path = self.log_dir / f"trades_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, 'w') as f:
                json.dump(self._trade_log, f, indent=2)
            logger.info(f"Trade log saved → {log_path}")
            print(f"\n  Trade log saved → {log_path}")
        except Exception as e:
            logger.warning(f"Could not save trade log: {e}")
