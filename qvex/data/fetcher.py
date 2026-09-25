"""
qvex/data/fetcher.py
=======================
Data fetching module for AIQuant.

Two data sources:
  1. CryptoDataDownload (CDD) — historical CSV for backtesting
     URL: https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_minute.csv
     No API key required. Full history up to July 2022 (~177MB per pair).
     NOTE: CDD files are static snapshots (not live-updated). The fetcher
     loads the last N days of whatever data is available in the file.
     Uses streaming tail-read — only downloads the required number of bars.

  2. Hyperliquid Public API — live 1m candles for paper/live trading
     Endpoint: https://api.hyperliquid.xyz/info  (type=candleSnapshot)
     No API key required for read-only market data.
     Returns: t, T, s, i, o, c, h, l, v, n fields.
"""

import time
import logging
import requests
import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
CDD_BASE_URL = "https://www.cryptodatadownload.com/cdd"
HL_INFO_URL  = "https://api.hyperliquid.xyz/info"
HL_MAX_BARS  = 5000

CDD_PAIR_MAP = {
    "BTCUSDT": "Binance_BTCUSDT_minute.csv",
    "ETHUSDT": "Binance_ETHUSDT_minute.csv",
    "SOLUSDT": "Binance_SOLUSDT_minute.csv",
    "BNBUSDT": "Binance_BNBUSDT_minute.csv",
    "XRPUSDT": "Binance_XRPUSDT_minute.csv",
}
HL_COIN_MAP = {
    "BTCUSDT": "BTC",
    "ETHUSDT": "ETH",
    "SOLUSDT": "SOL",
    "BNBUSDT": "BNB",
    "XRPUSDT": "XRP",
}


# ════════════════════════════════════════════════════════════════════════════
# BACKTEST DATA — CryptoDataDownload
# ════════════════════════════════════════════════════════════════════════════

def fetch_cdd_backtest(
    pair:      str  = "BTCUSDT",
    days:      int  = 90,
    cache_dir: Path = Path("data/raw"),
    force:     bool = False,
) -> pd.DataFrame:
    """
    Fetch historical 1m OHLCV data from CryptoDataDownload for backtesting.

    CDD files are static snapshots (newest data: ~July 2022 for BTC).
    The fetcher streams the tail of the file to get the last N days of
    available data — no need to download the full 177MB file.

    Parameters
    ----------
    pair      : Trading pair, e.g. 'BTCUSDT'
    days      : Number of days of history to load (from end of available data)
    cache_dir : Directory to cache parquet files
    force     : Force re-download even if cache exists

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    and a UTC DatetimeIndex named 'timestamp'.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pair}_1m_cdd.parquet"

    # Use cache if it exists and is not being forced
    if not force and cache_path.exists():
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:   # CDD is static — cache for 24h
            logger.info(f"Using cached CDD data ({cache_path.name}, {age_h:.1f}h old)")
            df = pd.read_parquet(cache_path)
            return df.iloc[-days * 1440:].copy()

    filename = CDD_PAIR_MAP.get(pair.upper())
    if not filename:
        raise ValueError(
            f"Pair '{pair}' not in CDD map. Available: {list(CDD_PAIR_MAP.keys())}"
        )

    url = f"{CDD_BASE_URL}/{filename}"
    logger.info(f"Streaming CDD data from {url} (last {days} days of available data)...")

    # CDD files are sorted newest-first after a 1-line comment + header.
    # Stream line-by-line and collect the required number of bars.
    bars_needed = days * 1440 + 500   # +500 for feature warmup
    raw_lines   = []
    header_line = ""
    skipped     = 0

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if skipped == 0:
                skipped = 1        # skip CDD comment row (https://www.CryptoDataDownload.com)
                continue
            if skipped == 1:
                header_line = line  # capture column header
                skipped = 2
                continue
            raw_lines.append(line)
            if len(raw_lines) >= bars_needed:
                break

    if not raw_lines:
        raise ValueError(
            f"No data received from CryptoDataDownload for {pair}. "
            "Check your internet connection or try again later."
        )

    # Parse collected lines into a DataFrame
    csv_text = header_line + "\n" + "\n".join(raw_lines)
    df_raw   = pd.read_csv(StringIO(csv_text))

    # Normalise column names
    # CDD columns: unix, date, symbol, open, high, low, close, Volume BTC, Volume USDT, tradecount
    col_map = {}
    for c in df_raw.columns:
        cl = c.lower().strip()
        if cl in ("unix", "timestamp"):
            col_map[c] = "unix"
        elif cl == "date":
            col_map[c] = "date"
        elif cl == "open":
            col_map[c] = "open"
        elif cl == "high":
            col_map[c] = "high"
        elif cl == "low":
            col_map[c] = "low"
        elif cl == "close":
            col_map[c] = "close"
        elif "volume" in cl and "usdt" not in cl:
            col_map[c] = "volume"
    df_raw = df_raw.rename(columns=col_map)

    # Build typed NumPy arrays
    open_arr  = df_raw["open"].to_numpy(dtype=np.float64)
    high_arr  = df_raw["high"].to_numpy(dtype=np.float64)
    low_arr   = df_raw["low"].to_numpy(dtype=np.float64)
    close_arr = df_raw["close"].to_numpy(dtype=np.float64)
    vol_arr   = df_raw["volume"].to_numpy(dtype=np.float64)

    if "unix" in df_raw.columns:
        ts_arr = df_raw["unix"].to_numpy(dtype=np.int64)
        index  = pd.to_datetime(ts_arr, unit="ms", utc=True)
    else:
        index  = pd.to_datetime(df_raw["date"], utc=True)

    df = pd.DataFrame({
        "open":   open_arr,
        "high":   high_arr,
        "low":    low_arr,
        "close":  close_arr,
        "volume": vol_arr,
    }, index=index)
    df.index.name = "timestamp"

    # CDD is newest-first — sort to chronological
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    # Cache full available dataset to parquet
    df.to_parquet(cache_path, compression="snappy")
    logger.info(f"CDD data cached: {len(df):,} bars → {cache_path}")

    # Return last N days of available data
    return df.iloc[-days * 1440:].copy()


# ════════════════════════════════════════════════════════════════════════════
# LIVE DATA — Hyperliquid Public API
# ════════════════════════════════════════════════════════════════════════════

def fetch_hyperliquid_candles(
    pair:     str = "BTCUSDT",
    n_bars:   int = 500,
    interval: str = "1m",
) -> pd.DataFrame:
    """
    Fetch recent candles from Hyperliquid's public API.
    No API key required.

    Parameters
    ----------
    pair     : Trading pair, e.g. 'BTCUSDT'
    n_bars   : Number of recent bars to fetch (max 5000)
    interval : Candle interval ('1m', '5m', '15m', '1h', '4h', '1d')

    Returns
    -------
    pd.DataFrame with columns [open, high, low, close, volume]
    and a UTC DatetimeIndex named 'timestamp'.
    """
    coin   = HL_COIN_MAP.get(pair.upper(), pair.replace("USDT", ""))
    n_bars = min(n_bars, HL_MAX_BARS)

    interval_ms = _interval_to_ms(interval)
    end_ms      = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms    = end_ms - (n_bars * interval_ms)

    all_bars = []
    since_ms = start_ms

    while since_ms < end_ms:
        batch_end = min(since_ms + HL_MAX_BARS * interval_ms, end_ms)
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin":      coin,
                "interval":  interval,
                "startTime": since_ms,
                "endTime":   batch_end,
            }
        }
        resp = requests.post(HL_INFO_URL, json=payload, timeout=15)
        resp.raise_for_status()
        bars = resp.json()

        if not bars or not isinstance(bars, list):
            break

        all_bars.extend(bars)
        since_ms = bars[-1]["T"] + 1
        if len(bars) < 10:
            break

    if not all_bars:
        raise ValueError(
            f"No candle data returned from Hyperliquid for {coin}. "
            "Check pair name and network connectivity."
        )

    ts_arr    = np.array([b["t"] for b in all_bars], dtype=np.int64)
    open_arr  = np.array([b["o"] for b in all_bars], dtype=np.float64)
    high_arr  = np.array([b["h"] for b in all_bars], dtype=np.float64)
    low_arr   = np.array([b["l"] for b in all_bars], dtype=np.float64)
    close_arr = np.array([b["c"] for b in all_bars], dtype=np.float64)
    vol_arr   = np.array([b["v"] for b in all_bars], dtype=np.float64)

    index = pd.to_datetime(ts_arr, unit="ms", utc=True)
    df = pd.DataFrame({
        "open":   open_arr,
        "high":   high_arr,
        "low":    low_arr,
        "close":  close_arr,
        "volume": vol_arr,
    }, index=index)
    df.index.name = "timestamp"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def fetch_latest_bar(pair: str = "BTCUSDT") -> Optional[pd.Series]:
    """Fetch the single most recent completed 1m bar from Hyperliquid."""
    try:
        df = fetch_hyperliquid_candles(pair=pair, n_bars=2, interval="1m")
        if len(df) >= 1:
            return df.iloc[-1]
    except Exception as e:
        logger.warning(f"fetch_latest_bar failed: {e}")
    return None


def get_available_pairs() -> list:
    """Return list of supported trading pairs."""
    return list(CDD_PAIR_MAP.keys())


# ── Helpers ──────────────────────────────────────────────────────────────────

def _interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds."""
    return {
        "1m": 60_000, "3m": 180_000, "5m": 300_000,
        "15m": 900_000, "30m": 1_800_000,
        "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }.get(interval, 60_000)