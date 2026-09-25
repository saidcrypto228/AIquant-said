from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .paper_trader import PaperTradingEngine
from .ml_live_trader import MLModelBundle

logger = logging.getLogger(__name__)


class MLPaperTrader:
    """Safe local paper-trading orchestrator for the trained ML ensemble."""

    def __init__(
        self,
        pair: str = "BTCUSDT",
        initial_capital: float = 100_000,
        kelly_fraction: float = 0.5,
        poll_interval_sec: float = 60,
        feature_window: int = 600,
        bundle_path: Optional[str] = None,
        log_dir: str = "logs/paper_trading",
    ):
        self.pair = pair.upper()
        self.feature_window = feature_window
        self.poll_interval_sec = poll_interval_sec
        self.tick_count = 0

        if self.pair != "BTCUSDT":
            raise ValueError("ML paper mode currently supports BTCUSDT only")

        if bundle_path is None:
            bundle_path = "models/ml_live_bundle.pkl"

        self.bundle_path = Path(bundle_path)
        self.model = MLModelBundle(str(self.bundle_path))

        self.engine = PaperTradingEngine(
            initial_capital=initial_capital,
            kelly_fraction=kelly_fraction,
            log_dir=log_dir,
        )

        self._warmup_history: Optional[pd.DataFrame] = None
        self._warmup_bars = 80_000

    def _load_warmup_history(self) -> pd.DataFrame:
        """Load enough local history for feature warm-up."""
        if self._warmup_history is not None:
            return self._warmup_history

        path = Path("data/raw/BTCUSDT_1m_full.parquet")

        if not path.exists():
            raise FileNotFoundError(
                f"ML paper warm-up dataset not found: {path}"
            )

        logger.info(
            "Loading local ML warm-up history | bars=%s",
            f"{self._warmup_bars:,}",
        )

        history = pd.read_parquet(path).tail(self._warmup_bars).copy()

        if not isinstance(history.index, pd.DatetimeIndex):
            if "timestamp" in history.columns:
                history["timestamp"] = pd.to_datetime(
                    history["timestamp"], utc=True
                )
                history = history.set_index("timestamp")

        history.index = pd.to_datetime(history.index, utc=True)
        history = history.sort_index()

        self._warmup_history = history

        logger.info(
            "Warm-up history ready | %s → %s | %s bars",
            history.index[0],
            history.index[-1],
            f"{len(history):,}",
        )

        return history

    def start(self, max_bars: Optional[int] = None):
        self.model.load()

        print("\n" + "=" * 70)
        print("  AIQuant ML PAPER TRADING")
        print("  EXECUTION: LOCAL SIMULATION ONLY")
        print(f"  Pair: {self.pair}")
        print(f"  Capital: ${self.engine.initial_capital:,.2f}")
        print(f"  Feature window: {self.feature_window}")
        print(f"  ML warm-up history: {self._warmup_bars:,} bars")
        print(f"  Model trained: {self.model.trained_at}")
        print(
            f"  Thresholds: LONG > {self.model.long_thresh} | "
            f"SHORT < {self.model.short_thresh}"
        )
        print("  Hyperliquid/private key: NOT USED")
        print("=" * 70)

        self.engine.run(
            signal_generator=self._signal_generator,
            max_bars=max_bars,
            poll_interval_sec=self.poll_interval_sec,
            lookback=self.feature_window,
            verbose=True,
        )

    def _signal_generator(self, df: pd.DataFrame) -> int:
        from ..features import build_full_feature_set

        self.tick_count += 1

        history = self._load_warmup_history()

        combined = pd.concat([history, df], axis=0)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()

        df_feat = build_full_feature_set(
            combined,
            verbose=False,
        )

        if df_feat.empty:
            logger.warning(
                "Paper ML tick %d: feature builder returned no rows; FLAT",
                self.tick_count,
            )
            return 0

        required = list(self.model.bundle['top_features'] if self.model.bundle else [])

        if not required:
            logger.warning(
                "Paper ML tick %d: model has no top_features; FLAT",
                self.tick_count,
            )
            return 0

        latest = df_feat.iloc[-1]

        missing = [
            name
            for name in required
            if name not in df_feat.columns or pd.isna(latest[name])
        ]

        if missing:
            logger.warning(
                "Paper ML tick %d: %d required ML features unavailable; FLAT",
                self.tick_count,
                len(missing),
            )
            return 0

        signal, score, breakdown = self.model.predict(df_feat)

        logger.info(
            "Paper ML tick %d | score=%+.4f | signal=%+d | "
            "xgb=%s | lgb=%s | lstm=%s",
            self.tick_count,
            score,
            signal,
            f"{breakdown.get('xgb', 0.0):+.4f}",
            f"{breakdown.get('lgb', 0.0):+.4f}",
            (
                f"{breakdown['lstm']:+.4f}"
                if breakdown.get("lstm") is not None
                else "None"
            ),
        )

        return int(signal)
