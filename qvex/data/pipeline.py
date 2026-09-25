"""
qvex/data/pipeline.py
========================
Master data pipeline orchestrator.
Fetches, merges, and prepares all data sources for feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, Dict

from .fetcher import BTCDataFetcher
from .onchain import OnChainDataFetcher

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Orchestrates the full data pipeline:
      1. OHLCV (1m primary + higher timeframes)
      2. Funding rates
      3. On-chain metrics (hash rate, mempool, etc.)
      4. Fear & Greed sentiment
      5. CoinGecko market data
    Merges everything into a single aligned DataFrame.
    """

    def __init__(self, data_dir: str = 'data', exchange_id: str = 'binance'):
        self.fetcher = BTCDataFetcher(exchange_id=exchange_id, data_dir=data_dir)
        self.onchain = OnChainDataFetcher(data_dir=data_dir)
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        symbol: str = 'BTC/USDT',
        since_date: str = '2024-01-01',
        fetch_onchain: bool = True,
        fetch_sentiment: bool = True,
        fetch_funding: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Run the full pipeline. Returns a merged, aligned DataFrame indexed at 1m.
        """
        logger.info("=" * 60)
        logger.info("Starting AIQuant Data Pipeline")
        logger.info("=" * 60)

        # ---- 1. Primary 1m OHLCV ----
        df_1m = self.fetcher.load_ohlcv(symbol, '1m')
        if df_1m is None or force_refresh:
            df_1m = self.fetcher.fetch_ohlcv(symbol, '1m', since_date)
        logger.info(f"1m OHLCV: {len(df_1m):,} bars")

        # ---- 2. Higher timeframe OHLCV (for multi-TF features) ----
        for tf in ['5m', '15m', '1h', '4h', '1d']:
            df_tf = self.fetcher.load_ohlcv(symbol, tf)
            if df_tf is None or force_refresh:
                df_tf = self.fetcher.fetch_ohlcv(symbol, tf, since_date)
            # Resample higher TF to 1m index via forward-fill
            df_tf_resampled = df_tf.reindex(df_1m.index, method='ffill')
            df_tf_resampled.columns = [f"{c}_{tf}" for c in df_tf_resampled.columns]
            df_1m = df_1m.join(df_tf_resampled, how='left')
            logger.info(f"Merged {tf} OHLCV into 1m frame")

        # ---- 3. Funding Rate ----
        if fetch_funding:
            df_fund = self.fetcher.fetch_funding_rates(symbol, since_date)
            if not df_fund.empty:
                df_fund_resampled = df_fund.reindex(df_1m.index, method='ffill')
                df_1m = df_1m.join(df_fund_resampled, how='left')
                logger.info(f"Merged funding rates ({len(df_fund):,} records)")

        # ---- 4. On-chain metrics ----
        if fetch_onchain:
            df_onchain = self.onchain.load_onchain()
            if df_onchain is None or force_refresh:
                df_onchain = self.onchain.fetch_all_onchain()
            if not df_onchain.empty:
                df_onchain_resampled = df_onchain.reindex(df_1m.index, method='ffill')
                df_1m = df_1m.join(df_onchain_resampled, how='left')
                logger.info(f"Merged on-chain metrics")

        # ---- 5. Fear & Greed ----
        if fetch_sentiment:
            df_fg = self.onchain.load_fear_greed()
            if df_fg is None or force_refresh:
                df_fg = self.onchain.fetch_fear_greed()
            if not df_fg.empty:
                df_fg_resampled = df_fg[['fear_greed']].reindex(df_1m.index, method='ffill')
                df_1m = df_1m.join(df_fg_resampled, how='left')
                logger.info(f"Merged Fear & Greed sentiment")

        # ---- 6. CoinGecko market data ----
        df_cg = self.onchain.load_coingecko()
        if df_cg is None or force_refresh:
            df_cg = self.onchain.fetch_coingecko_market_data()
        if not df_cg.empty:
            df_cg_resampled = df_cg[['market_cap', 'total_volume']].reindex(df_1m.index, method='ffill')
            df_1m = df_1m.join(df_cg_resampled, how='left')
            logger.info(f"Merged CoinGecko market data")

        # ---- Post-merge: enforce float64 and report NaN coverage ----
        # Convert all numeric columns to float64 in one pass using NumPy.
        # This avoids pandas per-column dtype promotion and ensures downstream
        # NumPy operations receive contiguous float64 arrays.
        numeric_cols = df_1m.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            df_1m[numeric_cols] = df_1m[numeric_cols].astype(np.float64)

        # Report NaN coverage per column using NumPy for speed
        nan_counts = np.sum(np.isnan(df_1m[numeric_cols].to_numpy()), axis=0)
        high_nan_cols = [
            f"{col}({int(n)})" for col, n in zip(numeric_cols, nan_counts) if n > 0
        ]
        if high_nan_cols:
            logger.info(f"NaN counts — {', '.join(high_nan_cols[:10])}{'...' if len(high_nan_cols) > 10 else ''}")

        # ---- Save merged dataset ----
        safe_symbol = symbol.replace('/', '_')
        out_path = self.processed_dir / f"{safe_symbol}_1m_merged.parquet"
        df_1m.to_parquet(out_path, compression='snappy')
        logger.info(f"Saved merged dataset: {df_1m.shape} → {out_path}")
        logger.info("=" * 60)
        logger.info("Data Pipeline Complete")
        logger.info("=" * 60)

        return df_1m

    def load_merged(self, symbol: str = 'BTC/USDT') -> Optional[pd.DataFrame]:
        """Load the pre-merged dataset."""
        safe_symbol = symbol.replace('/', '_')
        path = self.processed_dir / f"{safe_symbol}_1m_merged.parquet"
        if path.exists():
            logger.info(f"Loading merged dataset from {path}")
            return pd.read_parquet(path)
        return None