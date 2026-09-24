#!/usr/bin/env python3
"""
Институциональный движок факторов (Quant Factors Engine).
Реализует:
1. CS_ResMom_72h (Residual Momentum via OLS against BTC)
2. Dynamic Funding Z-Score
3. EVR Absorption Engine (Effort-Versus-Result with Wick Geometry)
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

class QuantFactorEngine:

    @staticmethod
    def compute_residual_momentum_72h(coin_closes: pd.Series, btc_closes: pd.Series) -> Tuple[float, float, float]:
        """
        Рассчитывает очищенный остаточный моментум (Residual Momentum).
        r_alt = alpha + beta * r_btc + epsilon
        Возвращает: (z_residual_mom, beta_btc, raw_rs_pct)
        """
        if len(coin_closes) < 70 or len(btc_closes) < 70:
            return 0.0, 1.0, 0.0

        # Совмещаем по минимальной длине
        n = min(len(coin_closes), len(btc_closes), 72)
        c_sub = coin_closes.iloc[-n:]
        b_sub = btc_closes.iloc[-n:]

        # Логарифмические часовые доходности
        r_alt = np.diff(np.log(c_sub.values))
        r_btc = np.diff(np.log(b_sub.values))

        if len(r_alt) < 20 or np.all(r_btc == 0):
            raw_rs = (math.log(c_sub.iloc[-1] / c_sub.iloc[0]) - math.log(b_sub.iloc[-1] / b_sub.iloc[0])) * 100
            return 0.0, 1.0, raw_rs

        # Скользящий OLS: оценка беты к BTC
        cov_mat = np.cov(r_alt, r_btc)
        var_btc = cov_mat[1, 1]
        cov_alt_btc = cov_mat[0, 1]

        beta = cov_alt_btc / (var_btc + 1e-9)
        beta = float(np.clip(beta, 0.1, 3.5))

        # Вектор остаточной доходности (идиосинкратическая альфа)
        residuals = r_alt - (beta * r_btc)
        cum_residual = float(np.sum(residuals))
        std_residual = float(np.std(residuals)) + 1e-7

        # Z-score остаточного моментума
        z_res_mom = cum_residual / (std_residual * math.sqrt(len(residuals)))

        # Сырая дельта доходностей для отображения в %
        raw_rs_pct = (math.log(c_sub.iloc[-1] / c_sub.iloc[0]) - math.log(b_sub.iloc[-1] / b_sub.iloc[0])) * 100

        return float(z_res_mom), float(beta), float(raw_rs_pct)

    @staticmethod
    @staticmethod
    def compute_funding_zscore(
        current_funding_apr: float,
        funding_history: list = None,
        clip_range: float = 3.0
    ) -> Tuple[float, bool]:
        """
        Институциональный расчет Z-score ставки финансирования (Аудит v9).
        Клиппинг на интервале [-clip_range, +clip_range].
        """
        if not math.isfinite(current_funding_apr):
            return 0.0, True

        # Если есть история от 3 измерений — считаем динамические выборочные статистики
        if funding_history and len(funding_history) >= 3:
            valid_h = [float(x) for x in funding_history if math.isfinite(x)]
            if len(valid_h) >= 3:
                arr = np.array(valid_h, dtype=float)
                mu = float(np.mean(arr))
                sigma = float(np.std(arr))
                if sigma < 1e-4:
                    sigma = 5.0  # Защита от нулевой дисперсии при стабильной ставке
            else:
                mu, sigma = 11.0, 25.0
        else:
            # Априорный байесовский базис (Perp Normal Prior: 11% APR, 25% std)
            mu, sigma = 11.0, 25.0

        raw_z = (current_funding_apr - mu) / sigma
        # Клиппинг для сохранения стационарности I(0)
        clipped_z = float(np.clip(raw_z, -clip_range, clip_range))

        # Сигнал перегрева: фандинг нейтрален, если Z находится в коридоре [-2.0, +2.0]
        is_funding_ok = abs(clipped_z) <= 2.0
        return clipped_z, is_funding_ok
    @staticmethod
    def evaluate_evr_absorption(
        open_px: float, high_px: float, low_px: float, close_px: float,
        volume: float, vol_sma: float, ema20_4h: float, atr_4h: float
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Оценивает качество поглощения объема через отношение Усилие/Результат (EVR)
        и микроструктурную геометрию свечи (Wick-to-Body Absorption Ratio).
        """
        total_range = max(high_px - low_px, atr_4h * 0.05, 1e-6)
        body = abs(close_px - open_px)
        lower_wick = min(open_px, close_px) - low_px

        # 1. Геометрия поглощения: доля выкупа нижнего фитиля
        absorption_geometry = (close_px - low_px) / total_range

        # 2. Усилие против результата (EVR)
        vol_effort = volume / max(vol_sma, 1e-6)
        price_result = total_range / max(atr_4h, 1e-6)
        evr_index = vol_effort / max(price_result, 0.2)

        # 3. Фильтр динамической поддержки: цена тестировала зону EMA20
        pullback_tested = (low_px <= ema20_4h * 1.010) and (close_px >= ema20_4h * 0.985)

        # 4. Комплексный балл институционального поглощения
        absorption_score = float(absorption_geometry * math.log(1.0 + max(evr_index, 0.0)))

        # Критерии для подтверждения чистого входа:
        # - Бычья свеча или длинный хвост снизу (absorption_geometry >= 0.55)
        # - Объем не ниже 85% от среднего (vol_effort >= 0.85)
        # - Тест EMA20
        # - Свеча закрылась выше середины диапазона
        mid_bar = (high_px + low_px) / 2.0
        is_absorption_confirmed = (
            pullback_tested and
            (close_px >= mid_bar) and
            (absorption_geometry >= 0.52) and
            (vol_effort >= 0.80) and
            (absorption_score >= 0.40)
        )

        metrics = {
            "absorption_geometry": round(absorption_geometry, 3),
            "evr_index": round(evr_index, 2),
            "vol_effort": round(vol_effort, 2),
            "absorption_score": round(absorption_score, 3),
            "pullback_tested": pullback_tested
        }

        return is_absorption_confirmed, absorption_score, metrics

    @staticmethod
    def verify_orderflow_confirmation(
        direction: str,
        cvd_ratio: float,
        cvd_usd: float,
        min_ratio_threshold: float = 0.15
    ) -> Tuple[bool, str]:
        """
        Верификация истинности импульса через дельту объемов (Фаза 3).
        Отсекает ложные пробои (Fakeouts) при расхождении хода цены и дельты.
        """
        dir_upper = direction.upper()

        if dir_upper == "LONG":
            # Истинный лонг: покупатели агрессивно выкупают аски
            if cvd_ratio >= min_ratio_threshold:
                return True, f"CONFIRMED_BUY (CVD Ratio: {cvd_ratio*100:+.1f}%)"
            elif cvd_ratio < 0:
                return False, f"REJECTED_DIVERGENCE (Цена растет, но CVD отрицательный: {cvd_ratio*100:+.1f}%)"
            else:
                return False, f"REJECTED_WEAK_DELTA (Недостаточная агрессия покупателей: {cvd_ratio*100:+.1f}%)"

        elif dir_upper == "SHORT":
            # Истинный шорт: продавцы бьют по бидам
            if cvd_ratio <= -min_ratio_threshold:
                return True, f"CONFIRMED_SELL (CVD Ratio: {cvd_ratio*100:+.1f}%)"
            elif cvd_ratio > 0:
                return False, f"REJECTED_DIVERGENCE (Цена падает, но CVD положительный: {cvd_ratio*100:+.1f}%)"
            else:
                return False, f"REJECTED_WEAK_DELTA (Недостаточная агрессия продавцов: {cvd_ratio*100:+.1f}%)"

        return False, "UNKNOWN_DIRECTION"