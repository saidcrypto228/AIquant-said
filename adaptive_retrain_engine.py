import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from quant_factors import QuantFactorEngine
from qvex_utils import atomic_write_json

logger = logging.getLogger("QVEX-Retrain")

class AdaptiveRetrainEngine:
    """
    Адаптивный контур обучения модели с институциональной очисткой 
    выборок: Triple-Barrier Purging + Embargo интервалы.
    """
    def __init__(self, target_r_multiple: float = 1.5, holding_horizon_bars: int = 6):
        self.r_mult = target_r_multiple
        self.horizon = holding_horizon_bars # Максимальное удержание барьера (24 часа = 6 баров)
        self.embargo_bars = 3              # Защитный зазор между фолдами (12 часов)
        self.model = LogisticRegression(C=0.1, penalty="l2", solver="liblinear", random_state=42)
        self.factor_engine = QuantFactorEngine()

    def apply_purging_and_embargo(self, df: pd.DataFrame, train_end_idx: int, test_start_idx: int) -> pd.DataFrame:
        """
        Удаление баров из train, чьи барьеры перекрывают test (Purging), 
        и добавление зазора безопасности (Embargo).
        """
        # Purge: отсекаем из обучения хвост, который заглядывает в тест
        safe_train_end = max(0, train_end_idx - self.horizon)
        train_indices = list(range(0, safe_train_end))

        # Embargo: сдвигаем начало теста вперед, чтобы избежать автокорреляции
        safe_test_start = test_start_idx + self.embargo_bars
        test_indices = list(range(safe_test_start, len(df)))

        return train_indices, test_indices

    def train_walk_forward(self, X: pd.DataFrame, y: pd.Series, split_ratio: float = 0.8) -> dict:
        """Обучение с гарантией чистоты OOS."""
        n = len(X)
        train_end = int(n * split_ratio)
        test_start = train_end

        train_idx, test_idx = self.apply_purging_and_embargo(X, train_end, test_start)

        X_train_raw = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test_raw = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Изолированное масштабирование
        X_train_scaled = self.factor_engine.fit_transform(X_train_raw)
        X_test_scaled = self.factor_engine.transform(X_test_raw)

        # Обучение модели
        self.model.fit(X_train_scaled, y_train)

        # Валидация на OOS
        preds = self.model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5

        logger.info(f"[ML-VALIDATION] OOS Clean AUC: {auc:.3f} (Purged samples: {self.horizon})")

        return {
            "auc": float(auc),
            "weights": self.model.coef_[0].tolist(),
            "intercept": float(self.model.intercept_[0]),
            "feature_names": list(X.columns)
        }

    def save_model_weights(self, meta: dict, path: Path | str = "data/meta_model.json"):
        """Атомарный сброс весов для мгновенного подхвата ядром."""
        atomic_write_json(path, meta)
        logger.info(f"✔ Адаптивная модель атомарно сохранена: {path}")
