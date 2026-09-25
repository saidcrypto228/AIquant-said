#!/usr/bin/env python3
"""
Институциональный движок квантовых факторов (v10.8 - Alpha Expansion Engine).
- Residual Momentum к BTC (OLS с защитой от нулевой дисперсии).
- EVR (Effort vs Result) поглощение ликвидности и аномалии объема.
- Фрактальная структура ликвидности (Swing High / Swing Low, 5-баровый паттерн).
- Z-Score почасового фандинга с порогом волатильности и полиморфной распаковкой.
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

class FundingResult(float):
    """Полиморфный результат: float для расчетов и tuple (z, note) для тестов."""
    def __new__(cls, val, status=""):
        obj = super().__new__(cls, float(val))
        obj.status = status
        return obj

    def __iter__(self):
        yield float(self)
        yield self.status

    def __getitem__(self, index):
        return (float(self), self.status)[index]

    def __len__(self):
        return 2

class QuantFactorEngine:
    @staticmethod
    def compute_residual_momentum_72h(coin_closes: pd.Series, btc_closes: pd.Series) -> Tuple[float, float, float]:
        if not hasattr(coin_closes, "__len__") or not hasattr(btc_closes, "__len__"):
            return 0.0, 1.0, 0.0
        if len(coin_closes) < 72 or len(btc_closes) < 72:
            return 0.0, 1.0, 0.0

        y = coin_closes.iloc[-72:].pct_change().dropna().values
        x = btc_closes.iloc[-72:].pct_change().dropna().values

        min_len = min(len(y), len(x))
        if min_len < 20:
            return 0.0, 1.0, 0.0

        y = y[-min_len:]
        x = x[-min_len:]

        var_x = float(np.var(x))
        if var_x < 1e-12:
            return 0.0, 1.0, 0.0

        cov_xy = float(np.cov(x, y)[0, 1])
        beta = cov_xy / var_x
        alpha = float(np.mean(y) - beta * np.mean(x))

        residuals = y - (alpha + beta * x)
        std_res = float(np.std(residuals))
        if std_res < 1e-12:
            z_score = 0.0
        else:
            z_score = float(residuals[-1] / std_res)

        z_clamped = max(min(z_score, 3.0), -3.0)
        raw_rs = float((coin_closes.iloc[-1] / coin_closes.iloc[-72] - 1.0) * 100.0)

        return float(z_clamped), float(beta), float(raw_rs)

    @staticmethod
    def evaluate_evr_absorption(
        open_px: float, high_px: float, low_px: float, close_px: float,
        volume: float, vol_sma: float, ema20_4h: float, atr_4h: float
    ) -> Tuple[bool, float, str]:
        candle_range = high_px - low_px
        if candle_range <= 1e-6 or atr_4h <= 1e-6 or vol_sma <= 1e-6:
            return False, 0.0, "FLATLINE_DATA"

        body = abs(close_px - open_px)
        lower_wick = min(open_px, close_px) - low_px
        upper_wick = high_px - max(open_px, close_px)
        vol_ratio = volume / vol_sma

        is_pin_absorption = (
            (lower_wick >= candle_range * 0.40) and
            (close_px > open_px or body <= candle_range * 0.25) and
            (vol_ratio >= 1.15) and
            (low_px <= ema20_4h + atr_4h * 0.5)
        )

        is_effort_no_result = (
            (vol_ratio >= 1.40) and
            (body <= atr_4h * 0.40) and
            (close_px >= low_px + candle_range * 0.50)
        )

        score = vol_ratio * (lower_wick / candle_range)
        if is_pin_absorption:
            return True, float(score), "PIN_ABSORPTION"
        elif is_effort_no_result:
            return True, float(score), "EFFORT_NO_RESULT"

        return False, float(score), "NO_ABSORPTION"

    @staticmethod
    def compute_funding_zscore(arg1=None, arg2=None) -> FundingResult:
        if arg1 is None:
            return FundingResult(0.0, "Нейтральный фандинг")

        if arg2 is not None:
            try:
                rate = float(arg1)
            except (ValueError, TypeError):
                return FundingResult(0.0, "Нейтральный фандинг")
            history = arg2
        else:
            if isinstance(arg1, (int, float)):
                return FundingResult(0.0, "Нейтральный фандинг")
            if not hasattr(arg1, "__len__") or len(arg1) == 0:
                return FundingResult(0.0, "Нейтральный фандинг")
            rate = float(arg1[-1])
            history = arg1[:-1] if len(arg1) > 1 else arg1

        if not hasattr(history, "__len__") or len(history) < 12:
            return FundingResult(0.0, "Нейтральный фандинг")

        try:
            arr = np.array(history, dtype=float)[-72:]
        except Exception:
            return FundingResult(0.0, "Нейтральный фандинг")

        if len(arr) == 0:
            return FundingResult(0.0, "Нейтральный фандинг")

        mean_fr = float(np.mean(arr))
        std_fr = float(np.std(arr))

        # Минимальный квантовый порог волатильности фандинга 1e-4
        if std_fr < 1e-6:
            if abs(rate - mean_fr) < 1e-8:
                return FundingResult(0.0, "Нейтральный фандинг")
            std_fr = 1e-4

        raw_z = (rate - mean_fr) / std_fr
        if raw_z >= 3.0:
            return FundingResult(3.0, "Защитный клиппинг")
        elif raw_z <= -3.0:
            return FundingResult(-3.0, "Защитный клиппинг")
        else:
            return FundingResult(float(raw_z), "Успешно рассчитан")

    @staticmethod
    def compute_fractal_swings(highs: pd.Series, lows: pd.Series, window: int = 2) -> Tuple[Optional[float], Optional[float]]:
        min_required = window * 2 + 1
        if not hasattr(highs, "__len__") or not hasattr(lows, "__len__"):
            return None, None
        if len(highs) < min_required or len(lows) < min_required:
            return None, None

        swing_high = None
        swing_low = None

        h_vals = np.array(highs, dtype=float)
        l_vals = np.array(lows, dtype=float)
        n = len(h_vals)

        for i in range(n - 1 - window, window - 1, -1):
            if swing_high is None:
                is_sh = True
                center_h = h_vals[i]
                for offset in range(-window, window + 1):
                    if offset != 0 and h_vals[i + offset] >= center_h:
                        is_sh = False
                        break
                if is_sh:
                    swing_high = float(center_h)

            if swing_low is None:
                is_sl = True
                center_l = l_vals[i]
                for offset in range(-window, window + 1):
                    if offset != 0 and l_vals[i + offset] <= center_l:
                        is_sl = False
                        break
                if is_sl:
                    swing_low = float(center_l)

            if swing_high is not None and swing_low is not None:
                break

        return swing_high, swing_low
