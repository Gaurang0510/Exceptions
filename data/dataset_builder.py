"""
Dataset Builder — Label Generation & Merging
=============================================
Merges financial news with stock price data and generates supervised
learning labels based on forward returns.

Label logic (configurable via ``config/settings.py``):

    return > +1 %  →  UP   (2)
    return < −1 %  →  DOWN (0)
    otherwise      →  NEUTRAL (1)
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd

from config.settings import (
    UP_THRESHOLD,
    DOWN_THRESHOLD,
    PREDICTION_HORIZON_HOURS,
    DEFAULT_TICKERS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)
from data.stock_data import fetch_yahoo_history
from data.news_data import load_kaggle_news, fetch_all_news

logger = logging.getLogger(__name__)

LABEL_MAP = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


# ────────────────────────────────────────────────────────────────────
#  Forward return computation
# ────────────────────────────────────────────────────────────────────
def compute_forward_return(
    prices: pd.DataFrame,
    horizon_days: int = 1,
) -> pd.DataFrame:
    """Compute forward percentage return over *horizon_days*.

    Expects ``prices`` to contain a ``Date`` and ``Close`` column.
    Returns the same frame with an additional ``forward_return`` column.
    """
    prices = prices.sort_values("Date").copy()
    # Handle both single-level and multi-level 'Close' column
    close_col = "Close"
    if close_col not in prices.columns:
        # Try to find a column that contains 'Close'
        close_candidates = [c for c in prices.columns if "close" in c.lower()]
        if close_candidates:
            close_col = close_candidates[0]
        else:
            raise KeyError("Cannot find a 'Close' price column.")

    prices["forward_return"] = (
        prices[close_col].shift(-horizon_days) / prices[close_col] - 1
    )
    return prices


def label_from_return(ret: float) -> int:
    """Map a forward return to an integer label."""
    if pd.isna(ret):
        return -1  # will be dropped
    if ret > UP_THRESHOLD:
        return 2  # UP
    elif ret < DOWN_THRESHOLD:
        return 0  # DOWN
    return 1  # NEUTRAL


# ────────────────────────────────────────────────────────────────────
#  Merge news + prices
# ────────────────────────────────────────────────────────────────────
def merge_news_with_prices(
    news: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Align news headlines with stock prices by date.

    Both frames are merged on a normalised date key.  The label for each
    news item is derived from the stock's forward return on the
    publication date.
    """
    if news.empty or prices.empty:
        logger.warning("Empty input frame for %s — skipping merge.", ticker)
        return pd.DataFrame()

    # Normalise date columns
    news = news.copy()
    if "published_at" in news.columns:
        news["merge_date"] = pd.to_datetime(
            news["published_at"], errors="coerce"
        ).dt.normalize()
    else:
        logger.warning("News frame has no 'published_at' column.")
        return pd.DataFrame()

    prices = compute_forward_return(prices, horizon_days=1)
    prices["merge_date"] = pd.to_datetime(prices["Date"], errors="coerce").dt.normalize()

    merged = pd.merge(news, prices, on="merge_date", how="inner")
    merged["label"] = merged["forward_return"].apply(label_from_return)
    merged = merged[merged["label"] >= 0].copy()
    merged["ticker"] = ticker
    merged["label_name"] = merged["label"].map(LABEL_MAP)

    return merged


# ────────────────────────────────────────────────────────────────────
#  Build complete dataset
# ────────────────────────────────────────────────────────────────────
def build_dataset(
    tickers: Optional[List[str]] = None,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    save: bool = True,
) -> pd.DataFrame:
    """End-to-end dataset construction.

    1. Fetch news (local CSV + APIs).
    2. Fetch stock prices.
    3. Merge & label.
    4. Optionally persist to ``data/processed/``.
    """
    tickers = tickers or DEFAULT_TICKERS
    all_news = fetch_all_news(tickers=tickers)

    frames = []
    for ticker in tickers:
        prices = fetch_yahoo_history(ticker, start=start, end=end)
        ticker_news = all_news[all_news.get("ticker", pd.Series()) == ticker] if "ticker" in all_news.columns else all_news
        merged = merge_news_with_prices(ticker_news, prices, ticker)
        if not merged.empty:
            frames.append(merged)

    if not frames:
        logger.warning("No labeled samples could be created.")
        return pd.DataFrame()

    dataset = pd.concat(frames, ignore_index=True)
    logger.info("Dataset built — %d samples.  Label distribution:\n%s",
                len(dataset), dataset["label_name"].value_counts().to_string())

    if save:
        out = PROCESSED_DATA_DIR / "labeled_dataset.parquet"
        dataset.to_parquet(out, index=False)
        logger.info("Saved to %s", out)

    return dataset


# ────────────────────────────────────────────────────────────────────
#  Quick-load helper
# ────────────────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    """Load the pre-built labeled dataset from disk."""
    path = PROCESSED_DATA_DIR / "labeled_dataset.parquet"
    if not path.exists():
        logger.info("Dataset not found — building now…")
        return build_dataset()
    return pd.read_parquet(path)


# ────────────────────────────────────────────────────────────────────
#  Synthetic / demo dataset for testing
# ────────────────────────────────────────────────────────────────────
def create_demo_dataset(n_samples: int = 2000, save: bool = True) -> pd.DataFrame:
    """Generate a synthetic dataset for demonstration & testing.

    Headlines carry inherent sentiment, and labels are generated to
    **correlate** with that sentiment so models learn a real signal.
    A noise factor (~10 %) keeps it realistic — not every bullish
    headline actually leads to a price increase.
    """
    rng = np.random.default_rng(42)

    # Templates grouped by directional bias - using strong signal words
    bullish_templates = [
        # Earnings & Performance - strong bullish keywords
        "{company} beats earnings estimates with record profit surge",
        "{company} exceeds analyst expectations as revenue soars",
        "{company} reports stellar quarterly results beating forecasts",
        "{company} outperforms rivals with impressive profit growth",
        "{company} crushes earnings estimates with blockbuster quarter",
        "{company} profit jumps on strong demand and rising margins",
        "{company} earnings surge past expectations boosting confidence",
        "{company} delivers outstanding results exceeding all targets",
        # Growth & Momentum
        "{company} revenue skyrockets as demand accelerates sharply",
        "{company} shares rally on bullish guidance and strong outlook",
        "{company} stock soars after upgrade from major analysts",
        "{company} gains momentum with accelerating growth trajectory",
        "{company} rebounds strongly with impressive recovery gains",
        "{company} expands rapidly with booming sales and profits",
        "{company} jumps on breakthrough product launch success",
        "{company} climbs on robust earnings and positive momentum",
        # Business Wins
        "Analysts upgrade {company} citing strong growth potential",
        "{company} announces massive buyback signaling confidence",
        "{company} dividend increase reflects bullish profit outlook",
        "{company} acquisition deal boosts growth and expands market",
        "{company} partnership drives innovation and revenue gains",
        "{company} investment in AI pays off with surging demand",
        "{company} dominates market with leading product success",
        "{company} achieves milestone with record-breaking sales",
        # Additional bullish
        "{company} thrives as profits surge past all expectations",
        "{company} flourishes with exceptional quarterly performance",
        "{company} succeeds in expansion delivering massive gains",
        "{company} wins big with outstanding market performance",
        "{company} positive outlook drives shares to new highs",
        "{company} optimistic guidance lifts stock on strong results",
    ]
    
    bearish_templates = [
        # Decline & Drop - strong bearish keywords
        "{company} stock plunges after disappointing earnings miss",
        "{company} shares crash on weak revenue and declining sales",
        "{company} tumbles on bearish guidance and falling margins",
        "{company} plummets after missing analyst estimates badly",
        "{company} sinks on concerns about slowing growth decline",
        "{company} drops sharply on disappointing quarterly loss",
        "{company} collapses after warning of significant shortfall",
        "{company} slumps on weak demand and shrinking profits",
        # Business Problems
        "{company} announces layoffs amid fears of recession",
        "{company} slashes workforce as growth stalls dramatically",
        "{company} cuts dividend signaling trouble and weakness",
        "{company} downgraded by analysts citing bearish concerns",
        "{company} warns of revenue decline and falling margins",
        "{company} struggles with declining sales and rising costs",
        "{company} fails to meet targets with disappointing results",
        "{company} loses market share amid fierce competition",
        # Crisis & Legal
        "{company} faces lawsuit over fraud allegations scandal",
        "{company} investigated for regulatory violations crisis",
        "{company} hit with penalty for compliance breach failure",
        "{company} scandal deepens with mounting losses and risk",
        "{company} recall crisis threatens product line collapse",
        "{company} bankruptcy fears grow as debt concerns mount",
        "{company} default risk rises on deteriorating finances",
        "{company} turmoil continues with executive departures",
        # Additional bearish
        "{company} pessimistic outlook sends shares tumbling",
        "{company} fear grips investors as losses accelerate",
        "{company} worst quarter in years drives selloff panic",
        "{company} headwinds intensify with worsening conditions",
        "{company} setback delays growth amid volatile markets",
        "{company} challenge mounts as problems pile up quickly",
    ]
    
    neutral_templates = [
        # Stability & Holding
        "{company} reports steady results unchanged from guidance",
        "{company} maintains stable outlook with flat growth",
        "{company} holds position with unchanged market share",
        "{company} remains steady amid sector consolidation",
        "{company} trading sideways as market awaits direction",
        "{company} stays flat with modest quarterly performance",
        "{company} keeps guidance unchanged for coming quarter",
        "{company} continues normal operations with stable results",
        # Mixed & Moderate
        "{company} reports mixed results inline with estimates",
        "{company} delivers moderate growth meeting expectations",
        "{company} balanced quarter with neutral market reaction",
        "{company} muted response to average earnings report",
        "{company} subdued trading as volume remains typical",
        "{company} quiet session with little movement expected",
        # Business as usual
        "{company} announces restructuring with transition ongoing",
        "{company} releases update maintaining current forecast",
        "{company} quarterly review shows typical performance",
        "{company} annual meeting concludes with no surprises",
        "{company} guidance remains in line with market estimate",
        "{company} sector outlook neutral amid ongoing evaluation",
    ]

    companies = ["Apple", "Google", "Microsoft", "Amazon", "Tesla",
                 "Meta", "NVIDIA", "Netflix", "JPMorgan", "Goldman Sachs"]
    tickers_map = {
        "Apple": "AAPL", "Google": "GOOGL", "Microsoft": "MSFT",
        "Amazon": "AMZN", "Tesla": "TSLA", "Meta": "META",
        "NVIDIA": "NVDA", "Netflix": "NFLX", "JPMorgan": "JPM",
        "Goldman Sachs": "GS",
    }

    NOISE_RATE = 0.10  # 10% of labels are flipped for realism (reduced from 15%)

    rows = []
    base_date = pd.Timestamp("2023-01-01")
    for i in range(n_samples):
        company = rng.choice(companies)

        # Pick a sentiment bucket roughly evenly
        bucket = rng.choice(["bullish", "bearish", "neutral"],
                            p=[0.40, 0.35, 0.25])
        if bucket == "bullish":
            headline = rng.choice(bullish_templates).format(company=company)
            forward_ret = abs(rng.normal(0.025, 0.012))   # positive
        elif bucket == "bearish":
            headline = rng.choice(bearish_templates).format(company=company)
            forward_ret = -abs(rng.normal(0.025, 0.012))  # negative
        else:
            headline = rng.choice(neutral_templates).format(company=company)
            forward_ret = rng.normal(0, 0.005)            # near-zero

        label = label_from_return(forward_ret)

        # Inject noise — randomly flip some labels
        if rng.random() < NOISE_RATE:
            label = int(rng.choice([0, 1, 2]))

        date = base_date + pd.Timedelta(days=int(rng.integers(0, 700)))
        rows.append(
            {
                "title": headline,
                "description": headline,
                "published_at": date,
                "ticker": tickers_map[company],
                "forward_return": forward_ret,
                "label": label,
                "label_name": LABEL_MAP[label],
                "Close": 100 + rng.normal(0, 20),
                "Volume": int(rng.integers(1_000_000, 50_000_000)),
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Demo dataset created — %d samples.", len(df))

    if save:
        out = RAW_DATA_DIR / "demo_dataset.csv"
        df.to_csv(out, index=False)
        out2 = PROCESSED_DATA_DIR / "labeled_dataset.parquet"
        df.to_parquet(out2, index=False)
        logger.info("Demo dataset saved to %s and %s", out, out2)

    return df
