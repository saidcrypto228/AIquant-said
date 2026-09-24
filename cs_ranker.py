#!/usr/bin/env python3
"""
Модернизированный кросс-секционный ранжировщик Фазы 2.
Использует очищенный OLS Residual Momentum и динамический Z-Score фандинга.
"""

import time
import math
import logging
from typing import List, Dict, Any
import pandas as pd
from hyperliquid.info import Info
from quant_factors import QuantFactorEngine

logger = logging.getLogger("CSRanker")

class CrossSectionalRanker:
    def __init__(self, info: Info, coins: List[str]):
        self.info = info
        self.coins = coins

    def fetch_72h_series(self) -> Dict[str, pd.Series]:
        """Собирает часовые ряды цен закрытия за 72 часа для всей корзины и BTC."""
        series_map = {}
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (76 * 3600 * 1000)
        all_symbols = list(set(self.coins + ["BTC"]))

        for sym in all_symbols:
            try:
                raw = self.info.candles_snapshot(
                    name=sym,
                    interval="1h",
                    startTime=start_ms,
                    endTime=end_ms
                )
                if not raw or len(raw) < 70:
                    continue

                df = pd.DataFrame(raw).sort_values("t").reset_index(drop=True)
                series_map[sym] = df["c"].astype(float)
            except Exception as e:
                logger.debug(f"Ошибка получения свечей {sym}: {e}")

        return series_map

    def get_live_funding_map(self) -> Dict[str, float]:
        funding_map = {}
        try:
            data = self.info.meta_and_asset_ctxs()
            universe = data[0]["universe"]
            asset_ctxs = data[1]
            for idx, asset in enumerate(universe):
                coin_name = asset["name"]
                if coin_name in self.coins:
                    hourly_f = float(asset_ctxs[idx].get("funding", 0.0))
                    funding_map[coin_name] = hourly_f * 24 * 365 * 100
        except Exception:
            pass
        return funding_map

    def rank_universe(self, max_funding_apr: float = 45.0) -> pd.DataFrame:
        series_map = self.fetch_72h_series()
        if "BTC" not in series_map:
            return pd.DataFrame()

        btc_series = series_map["BTC"]
        funding_map = self.get_live_funding_map()

        records = []
        for coin in self.coins:
            if coin not in series_map:
                continue

            coin_series = series_map[coin]
            z_res_mom, beta_btc, raw_rs_pct = QuantFactorEngine.compute_residual_momentum_72h(coin_series, btc_series)
            f_apr = funding_map.get(coin, 0.0)
            z_fr, is_funding_ok = QuantFactorEngine.compute_funding_zscore(f_apr)

            # Квантовый фильтр отбора:
            # 1. Z-score остаточного моментума >= 0.5 (статистически опережает рынок сверх своей беты)
            # 2. Фандинг в пределах допустимого Z-score
            # 3. Сырой RS >= 1.5%
            is_alpha_valid = (z_res_mom >= 0.35) and (raw_rs_pct >= 1.5)

            if not is_funding_ok or f_apr > max_funding_apr:
                status = "REJECTED (ПЕРЕГРЕВ ФАНДИНГА)"
            elif not is_alpha_valid:
                status = "REJECTED (НЕТ ОСТАТОЧНОЙ АЛЬФЫ)"
            else:
                status = "QUALIFIED (ГОТОВ К ВХОДУ)"

            records.append({
                "coin": coin,
                "z_res_mom": round(z_res_mom, 2),
                "beta_btc": round(beta_btc, 2),
                "rs_to_btc_pct": round(raw_rs_pct, 2),
                "funding_apr": round(f_apr, 1),
                "z_funding": round(z_fr, 2),
                "status": status,
                "quant_score": round(z_res_mom * (1.0 / max(1.0, beta_btc)), 3) if (is_funding_ok and is_alpha_valid) else -999.0
            })

        df_rank = pd.DataFrame(records)
        if df_rank.empty:
            return df_rank

        # Сортировка по квантовой оценке остаточного моментума
        df_rank = df_rank.sort_values("quant_score", ascending=False).reset_index(drop=True)
        df_rank["rank"] = df_rank.index + 1
        return df_rank
