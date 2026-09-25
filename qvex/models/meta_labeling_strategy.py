"""
Meta-Labeling Institutional Strategy Module for Prop Trading.
Architecture:
  - Timeframe: 1H
  - Primary Model: Trend Pullback Heuristic (EMA100 Regime + EMA25/RSI Reversion)
  - Secondary Model: XGBoost Meta-Label Classifier (Filter false pullbacks)
  - Risk Management: Dynamic sizing (0.4% - 0.75%), Monoposition, Daily Circuit Breaker (-2%)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from dataclasses import dataclass


@dataclass
class MetaStrategyConfig:
    ema_fast: int = 25
    ema_slow: int = 100
    rsi_threshold: float = 42.0
    sl_mult: float = 1.2
    tp_mult: float = 1.8
    timeout_bars: int = 20
    enter_prob: float = 0.55
    risk_low: float = 0.0040
    risk_high: float = 0.0075
    daily_loss_limit_usd: float = 200.0
    fee_pct: float = 0.0012


class MetaLabelingStrategy:
    def __init__(self, cfg: MetaStrategyConfig = MetaStrategyConfig()):
        self.cfg = cfg
        self.model = None

    def generate_primary_signals(self, df: pd.DataFrame) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)

        ema_25 = pd.Series(c).ewm(span=self.cfg.ema_fast, adjust=False).mean().to_numpy()
        ema_100 = pd.Series(c).ewm(span=self.cfg.ema_slow, adjust=False).mean().to_numpy()

        rsi_norm = df["rsi_norm"].to_numpy() if "rsi_norm" in df.columns else np.zeros(len(df))
        rsi = (rsi_norm * 50.0) + 50.0

        macro_bull = c > ema_100
        pullback = (rsi < self.cfg.rsi_threshold) | (l <= ema_25)
        return macro_bull & pullback

    def label_setups(self, df: pd.DataFrame, primary_signals: np.ndarray) -> np.ndarray:
        c = df["close"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        atr = df["atr_fast"].to_numpy(dtype=np.float64)
        n = len(df)

        meta_labels = np.full(n, np.nan)
        for i in range(n - self.cfg.timeout_bars):
            if not primary_signals[i]:
                continue
            entry = c[i]
            cur_atr = atr[i]
            sl_price = entry - (cur_atr * self.cfg.sl_mult)
            tp_price = entry + (cur_atr * self.cfg.tp_mult)

            success = 0
            for k in range(1, self.cfg.timeout_bars + 1):
                if l[i + k] <= sl_price:
                    success = 0
                    break
                if h[i + k] >= tp_price:
                    success = 1
                    break
            meta_labels[i] = success
        return meta_labels

    def train_meta_model(self, X_train: np.ndarray, y_train: np.ndarray):
        pos_count = sum(y_train)
        neg_count = len(y_train) - pos_count
        scale_pos = (neg_count / pos_count) if pos_count > 0 else 1.0

        self.model = xgb.XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos,
            random_state=42,
            tree_method="hist"
        )
        self.model.fit(X_train, y_train)

    def predict_probs(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        return self.model.predict_proba(X)[:, 1]
