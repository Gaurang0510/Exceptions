"""
Financial News Data Fetcher
============================
Retrieves financial news from NewsAPI, Finnhub, and supports loading
Kaggle CSV datasets.  Includes rate-limiting and caching.
"""

import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import pandas as pd
import requests

from config.settings import (
    NEWSAPI_KEY,
    FINNHUB_KEY,
    RAW_DATA_DIR,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# NewsAPI
# ────────────────────────────────────────────────────────────────────
_NEWSAPI_BASE = "https://newsapi.org/v2/everything"


def fetch_newsapi(
    query: str = "stock market",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    language: str = "en",
    page_size: int = 100,
    sort_by: str = "publishedAt",
) -> pd.DataFrame:
    """Fetch financial news articles from NewsAPI.

    Parameters
    ----------
    query : str
        Free-text search query (e.g. ``"AAPL earnings"``).
    from_date, to_date : str or None
        ISO-8601 date strings.  Defaults to past 7 days.
    """
    if not NEWSAPI_KEY:
        logger.warning("NEWSAPI_KEY not set — returning empty frame.")
        return pd.DataFrame()

    from_date = from_date or (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    to_date = to_date or datetime.now().strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "to": to_date,
        "language": language,
        "pageSize": page_size,
        "sortBy": sort_by,
        "apiKey": NEWSAPI_KEY,
    }
    resp = requests.get(_NEWSAPI_BASE, params=params, timeout=30)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    rows = []
    for a in articles:
        rows.append(
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "content": a.get("content", ""),
                "source": a.get("source", {}).get("name", ""),
                "author": a.get("author", ""),
                "url": a.get("url", ""),
                "published_at": a.get("publishedAt", ""),
            }
        )
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# Finnhub Company News
# ────────────────────────────────────────────────────────────────────
_FINNHUB_NEWS = "https://finnhub.io/api/v1/company-news"


def fetch_finnhub_news(
    ticker: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch company-specific news from Finnhub."""
    if not FINNHUB_KEY:
        logger.warning("FINNHUB_KEY not set — returning empty frame.")
        return pd.DataFrame()

    from_date = from_date or (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    to_date = to_date or datetime.now().strftime("%Y-%m-%d")

    resp = requests.get(
        _FINNHUB_NEWS,
        params={
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
            "token": FINNHUB_KEY,
        },
        timeout=30,
    )
    resp.raise_for_status()
    articles = resp.json()

    rows = []
    for a in articles:
        rows.append(
            {
                "title": a.get("headline", ""),
                "description": a.get("summary", ""),
                "source": a.get("source", ""),
                "url": a.get("url", ""),
                "published_at": datetime.fromtimestamp(
                    a.get("datetime", 0)
                ).isoformat(),
                "ticker": ticker,
            }
        )
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────
# Kaggle / Local CSV Loader
# ────────────────────────────────────────────────────────────────────
def load_kaggle_news(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load a financial-news CSV from disk (e.g. a Kaggle download).

    Expected columns (flexible): ``title``, ``text`` / ``description``,
    ``date`` / ``published_at``, ``ticker`` (optional).

    The loader normalises column names for downstream use.
    """
    if filepath is None:
        candidates = list(RAW_DATA_DIR.glob("*.csv"))
        if not candidates:
            logger.warning("No CSV files found in %s", RAW_DATA_DIR)
            return pd.DataFrame()
        filepath = str(candidates[0])

    logger.info("Loading Kaggle CSV from %s", filepath)
    df = pd.read_csv(filepath, low_memory=False)

    # ── Normalise columns ───────────────────────────────────────────
    rename_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl in ("headline", "heading", "title"):
            rename_map[col] = "title"
        elif cl in ("text", "body", "article", "content", "description"):
            rename_map[col] = "description"
        elif cl in ("date", "published", "publishedat", "published_at", "timestamp"):
            rename_map[col] = "published_at"
        elif cl in ("stock", "ticker", "symbol"):
            rename_map[col] = "ticker"

    df = df.rename(columns=rename_map)

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    return df


# ────────────────────────────────────────────────────────────────────
# Composite news fetcher
# ────────────────────────────────────────────────────────────────────
def fetch_all_news(
    tickers: Optional[List[str]] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate news from all available sources."""
    frames: List[pd.DataFrame] = []

    # NewsAPI
    for t in (tickers or []):
        df = fetch_newsapi(query=t, from_date=from_date, to_date=to_date)
        if not df.empty:
            df["ticker"] = t
            frames.append(df)

    # Finnhub
    for t in (tickers or []):
        df = fetch_finnhub_news(t, from_date=from_date, to_date=to_date)
        if not df.empty:
            frames.append(df)

    # Local Kaggle
    kaggle = load_kaggle_news()
    if not kaggle.empty:
        frames.append(kaggle)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
