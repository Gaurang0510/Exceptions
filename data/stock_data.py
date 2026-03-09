"""
Stock Price Data Fetcher
========================
Retrieves historical and real-time stock price data from Yahoo Finance,
Alpha Vantage, and Finnhub.  Includes rate-limiting, caching, and
error-handling utilities.
"""

import time
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from config.settings import (
    ALPHA_VANTAGE_KEY,
    FINNHUB_KEY,
    CACHE_DIR,
    DEFAULT_TICKERS,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Disk-based cache helper
# ────────────────────────────────────────────────────────────────────
def _cache_path(key: str) -> Path:
    hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{hashed}.parquet"


def _read_cache(key: str, max_age_hours: int = 6) -> Optional[pd.DataFrame]:
    path = _cache_path(key)
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < max_age_hours * 3600:
            logger.debug("Cache hit for key=%s", key)
            return pd.read_parquet(path)
    return None


def _write_cache(key: str, df: pd.DataFrame) -> None:
    df.to_parquet(_cache_path(key))


# ────────────────────────────────────────────────────────────────────
# Yahoo Finance
# ────────────────────────────────────────────────────────────────────
def fetch_yahoo_history(
    ticker: str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV history via ``yfinance``."""
    end = end or datetime.now().strftime("%Y-%m-%d")
    cache_key = f"yahoo_{ticker}_{start}_{end}_{interval}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    logger.info("Fetching Yahoo Finance data: %s [%s → %s]", ticker, start, end)
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        logger.warning("No data returned for %s", ticker)
        return df

    df = df.reset_index()
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip("_") for col in df.columns]

    _write_cache(cache_key, df)
    return df


def fetch_yahoo_realtime(ticker: str) -> dict:
    """Get latest quote snapshot via ``yfinance``."""
    t = yf.Ticker(ticker)
    info = t.info
    return {
        "ticker": ticker,
        "price": info.get("regularMarketPrice"),
        "change": info.get("regularMarketChange"),
        "change_pct": info.get("regularMarketChangePercent"),
        "volume": info.get("regularMarketVolume"),
        "timestamp": datetime.now().isoformat(),
    }


# ────────────────────────────────────────────────────────────────────
# Alpha Vantage
# ────────────────────────────────────────────────────────────────────
_AV_BASE = "https://www.alphavantage.co/query"
_AV_RATE_LIMIT = 12  # free-tier: 5 calls / min – we space conservatively


def fetch_alpha_vantage_daily(ticker: str) -> pd.DataFrame:
    """Daily adjusted close from Alpha Vantage (free tier)."""
    if not ALPHA_VANTAGE_KEY:
        logger.warning("ALPHA_VANTAGE_KEY not set — skipping.")
        return pd.DataFrame()

    cache_key = f"av_daily_{ticker}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": ticker,
        "outputsize": "full",
        "apikey": ALPHA_VANTAGE_KEY,
    }
    resp = requests.get(_AV_BASE, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    ts = data.get("Time Series (Daily)", {})
    if not ts:
        logger.warning("Alpha Vantage returned empty payload for %s", ticker)
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(ts, orient="index").astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df.columns = [c.split(". ")[1] if ". " in c else c for c in df.columns]
    df = df.reset_index().rename(columns={"index": "Date"})

    _write_cache(cache_key, df)
    time.sleep(_AV_RATE_LIMIT)  # respect rate limit
    return df


# ────────────────────────────────────────────────────────────────────
# Finnhub
# ────────────────────────────────────────────────────────────────────
_FINNHUB_BASE = "https://finnhub.io/api/v1"


def fetch_finnhub_quote(ticker: str) -> dict:
    """Real-time quote from Finnhub."""
    if not FINNHUB_KEY:
        logger.warning("FINNHUB_KEY not set — skipping.")
        return {}

    resp = requests.get(
        f"{_FINNHUB_BASE}/quote",
        params={"symbol": ticker, "token": FINNHUB_KEY},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "ticker": ticker,
        "current": data.get("c"),
        "high": data.get("h"),
        "low": data.get("l"),
        "open": data.get("o"),
        "prev_close": data.get("pc"),
        "timestamp": datetime.fromtimestamp(data.get("t", 0)).isoformat(),
    }


# ────────────────────────────────────────────────────────────────────
# Batch helpers
# ────────────────────────────────────────────────────────────────────
def fetch_multiple_tickers(
    tickers: Optional[list] = None,
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Download history for a list of tickers and return a combined frame."""
    tickers = tickers or DEFAULT_TICKERS
    frames = []
    for t in tickers:
        df = fetch_yahoo_history(t, start=start, end=end)
        if not df.empty:
            df["Ticker"] = t
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
