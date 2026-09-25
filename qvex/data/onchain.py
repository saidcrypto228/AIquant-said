"""
qvex/data/onchain.py
=======================
On-chain and sentiment data ingestion using free public APIs.

Sources:
  - alternative.me  : Crypto Fear & Greed Index (free, no key)
  - blockchain.info : Bitcoin hash rate, mempool, transaction volume (free)
  - Coindesk / CoinGecko: Market cap, dominance (free)
  - BGeometrics     : MVRV, SOPR proxies (free)
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class OnChainDataFetcher:
    """
    Fetches on-chain and sentiment data from free public APIs.
    Designed to be merged with OHLCV data for feature engineering.
    """

    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    BLOCKCHAIN_INFO_URL = "https://api.blockchain.info/charts/{metric}"
    COINGECKO_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, data_dir: str = 'data'):
        self.data_dir = Path(data_dir)
        self.external_dir = self.data_dir / 'external'
        self.external_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fear & Greed Index
    # ------------------------------------------------------------------

    def fetch_fear_greed(self, limit: int = 2000, save: bool = True) -> pd.DataFrame:
        """
        Fetch the Crypto Fear & Greed Index from alternative.me.
        Returns daily sentiment scores (0=Extreme Fear, 100=Extreme Greed).
        """
        logger.info("Fetching Fear & Greed Index from alternative.me...")
        try:
            resp = requests.get(self.FEAR_GREED_URL, params={'limit': limit, 'format': 'json'}, timeout=30)
            resp.raise_for_status()
            data = resp.json()['data']
            df = pd.DataFrame([{
                'timestamp': pd.to_datetime(int(d['timestamp']), unit='s', utc=True),
                'fear_greed': int(d['value']),
                'fear_greed_class': d['value_classification'],
            } for d in data])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            if save:
                path = self.external_dir / 'fear_greed.parquet'
                df.to_parquet(path)
                logger.info(f"Saved Fear & Greed data to {path}")
            return df
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Blockchain.info Metrics
    # ------------------------------------------------------------------

    def fetch_blockchain_metric(
        self,
        metric: str,
        timespan: str = '2years',
        rolling_average: str = '8hours',
    ) -> pd.DataFrame:
        """
        Fetch a metric from blockchain.info.
        Available metrics: hash-rate, n-transactions, mempool-size,
                           estimated-transaction-volume-usd, miners-revenue,
                           transaction-fees-usd, avg-block-size
        """
        url = self.BLOCKCHAIN_INFO_URL.format(metric=metric)
        params = {
            'timespan': timespan,
            'rollingAverage': rolling_average,
            'format': 'json',
            'sampled': 'true',
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            values = resp.json()['values']
            df = pd.DataFrame(values)
            df['timestamp'] = pd.to_datetime(df['x'], unit='s', utc=True)
            df.rename(columns={'y': metric.replace('-', '_')}, inplace=True)
            df.set_index('timestamp', inplace=True)
            df.drop(columns=['x'], inplace=True)
            logger.info(f"Fetched {len(df):,} records for blockchain metric: {metric}")
            return df
        except Exception as e:
            logger.error(f"Error fetching blockchain metric {metric}: {e}")
            return pd.DataFrame()

    def fetch_all_onchain(self, save: bool = True) -> pd.DataFrame:
        """
        Fetch and merge all key on-chain metrics into a single DataFrame.
        """
        logger.info("Fetching all on-chain metrics...")
        metrics = [
            'hash-rate',
            'n-transactions',
            'mempool-size',
            'estimated-transaction-volume-usd',
            'miners-revenue',
            'transaction-fees-usd',
        ]
        dfs = []
        for m in metrics:
            df = self.fetch_blockchain_metric(m)
            if not df.empty:
                dfs.append(df)
            time.sleep(1.5)  # Respectful rate limiting

        if not dfs:
            logger.warning("No on-chain data fetched.")
            return pd.DataFrame()

        combined = dfs[0]
        for df in dfs[1:]:
            combined = combined.join(df, how='outer')

        combined.sort_index(inplace=True)
        combined = combined.ffill()  # Forward-fill gaps

        if save:
            path = self.external_dir / 'onchain_metrics.parquet'
            combined.to_parquet(path)
            logger.info(f"Saved on-chain metrics to {path}")

        return combined

    # ------------------------------------------------------------------
    # CoinGecko — Market Cap, Dominance, Volume
    # ------------------------------------------------------------------

    def fetch_coingecko_market_data(
        self,
        coin_id: str = 'bitcoin',
        vs_currency: str = 'usd',
        days: int = 365,
        save: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch daily market cap, total volume, and price from CoinGecko (free tier).
        """
        logger.info(f"Fetching CoinGecko market data for {coin_id}...")
        url = f"{self.COINGECKO_URL}/coins/{coin_id}/market_chart"
        params = {'vs_currency': vs_currency, 'days': days, 'interval': 'daily'}
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            prices = pd.DataFrame(data['prices'], columns=['ts', 'cg_price'])
            mcap = pd.DataFrame(data['market_caps'], columns=['ts', 'market_cap'])
            vol = pd.DataFrame(data['total_volumes'], columns=['ts', 'total_volume'])

            df = prices.merge(mcap, on='ts').merge(vol, on='ts')
            df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            df.drop(columns=['ts'], inplace=True)

            if save:
                path = self.external_dir / f'coingecko_{coin_id}.parquet'
                df.to_parquet(path)
                logger.info(f"Saved CoinGecko data to {path}")
            return df
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Loader helpers
    # ------------------------------------------------------------------

    def load_fear_greed(self) -> Optional[pd.DataFrame]:
        path = self.external_dir / 'fear_greed.parquet'
        return pd.read_parquet(path) if path.exists() else None

    def load_onchain(self) -> Optional[pd.DataFrame]:
        path = self.external_dir / 'onchain_metrics.parquet'
        return pd.read_parquet(path) if path.exists() else None

    def load_coingecko(self, coin_id: str = 'bitcoin') -> Optional[pd.DataFrame]:
        path = self.external_dir / f'coingecko_{coin_id}.parquet'
        return pd.read_parquet(path) if path.exists() else None