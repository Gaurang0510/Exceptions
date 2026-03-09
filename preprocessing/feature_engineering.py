"""
Feature Engineering Module
===========================
Constructs five categories of features from preprocessed data:

1. **Text features** — TF-IDF, n-grams, sentence embeddings
2. **Sentiment features** — TextBlob, VADER, keyword-based signals
3. **Market features** — volatility, moving averages, momentum, volume
4. **Temporal features** — hour-of-day, day-of-week, market session
5. **Dense embeddings** — MiniLM sentence embeddings (384-dim)

All feature builders return NumPy arrays or DataFrames that can be
concatenated horizontally before model training.
"""

import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from config.settings import (
    TFIDF_MAX_FEATURES,
    NGRAM_RANGE,
    MOVING_AVERAGE_WINDOWS,
    VOLATILITY_WINDOW,
    MOMENTUM_WINDOW,
    SENTIMENT_MODEL,
    MAX_SEQUENCE_LENGTH,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. TEXT FEATURES
# ═══════════════════════════════════════════════════════════════════
class TextFeatureExtractor:
    """TF-IDF + n-gram feature extractor with fit / transform API."""

    def __init__(
        self,
        max_features: int = TFIDF_MAX_FEATURES,
        ngram_range: tuple = NGRAM_RANGE,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=2,
        )

    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        return self.vectorizer.fit_transform(texts.fillna("")).toarray()

    def transform(self, texts: pd.Series) -> np.ndarray:
        return self.vectorizer.transform(texts.fillna("")).toarray()


# ═══════════════════════════════════════════════════════════════════
# 1b. SENTENCE EMBEDDINGS (MiniLM - fast, 384-dim)
# ═══════════════════════════════════════════════════════════════════
_SENTENCE_MODEL = None
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


def _load_sentence_model():
    """Lazy-load the sentence transformer model."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence embedding model: all-MiniLM-L6-v2")
            # Force CPU for sentence-transformers (PyTorch doesn't support RTX 50-series yet)
            _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        except Exception as exc:
            logger.warning("Could not load sentence model: %s", exc)
            _SENTENCE_MODEL = False
    return _SENTENCE_MODEL


def get_sentence_embeddings(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """Get dense sentence embeddings using MiniLM.
    
    Returns shape (n_texts, 384) - captures semantic meaning of headlines.
    """
    model = _load_sentence_model()
    if not model:
        logger.warning("Sentence model unavailable — returning zeros.")
        return np.zeros((len(texts), EMBEDDING_DIM))
    
    # Ensure texts are strings
    texts = [str(t) if t else "" for t in texts]
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings


def get_single_embedding(text: str) -> np.ndarray:
    """Get embedding for a single text (1, 384)."""
    model = _load_sentence_model()
    if not model:
        return np.zeros((1, EMBEDDING_DIM))
    return model.encode([str(text)], show_progress_bar=False)


# ── FinBERT embeddings (lazy-loaded) ──────────────────────────────
_FINBERT_MODEL = None
_FINBERT_TOKENIZER = None


def _load_finbert():
    global _FINBERT_MODEL, _FINBERT_TOKENIZER
    if _FINBERT_MODEL is None:
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch

            logger.info("Loading FinBERT model: %s", SENTIMENT_MODEL)
            _FINBERT_TOKENIZER = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
            _FINBERT_MODEL = AutoModel.from_pretrained(SENTIMENT_MODEL)
            _FINBERT_MODEL.eval()
        except Exception as exc:
            logger.warning("Could not load FinBERT: %s", exc)
            _FINBERT_MODEL = False
    return _FINBERT_MODEL, _FINBERT_TOKENIZER


def get_finbert_embeddings(texts: list, batch_size: int = 32) -> np.ndarray:
    """Extract [CLS] embeddings from FinBERT for a list of texts."""
    import torch

    model, tokenizer = _load_finbert()
    if not model:
        logger.warning("FinBERT unavailable — returning zeros.")
        return np.zeros((len(texts), 768))

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :].numpy()
        all_embeddings.append(cls_emb)

    return np.vstack(all_embeddings)


# ═══════════════════════════════════════════════════════════════════
# 2. SENTIMENT FEATURES
# ═══════════════════════════════════════════════════════════════════
# Massively expanded keyword dictionaries for robust sentiment detection
BULLISH_KEYWORDS = {
    # Earnings & Performance
    "beat", "beats", "beating", "record", "exceeded", "exceeds", "exceeding",
    "outperform", "outperforms", "outperformed", "outperforming", "stellar",
    "blockbuster", "impressive", "robust", "strong", "stronger", "strongest",
    "profit", "profits", "profitable", "profitability", "earnings",
    # Growth indicators
    "surge", "surges", "surging", "surged", "soar", "soars", "soaring", "soared",
    "rally", "rallies", "rallying", "rallied", "jump", "jumps", "jumping", "jumped",
    "spike", "spikes", "spiking", "spiked", "boom", "booms", "booming", "boomed",
    "skyrocket", "skyrockets", "skyrocketing", "breakthrough", "breakout",
    "gain", "gains", "gaining", "gained", "rise", "rises", "rising", "rose",
    "climb", "climbs", "climbing", "climbed", "advance", "advances", "advancing",
    # Positive actions
    "upgrade", "upgraded", "upgrades", "upgrading", "buy", "buying", "buyback",
    "boost", "boosts", "boosting", "boosted", "lift", "lifts", "lifting", "lifted",
    "expand", "expands", "expanding", "expanded", "expansion", "grow", "grows",
    "growing", "grew", "growth", "accelerate", "accelerates", "accelerating",
    # Business positive
    "innovation", "innovative", "innovates", "partnership", "partnerships",
    "acquisition", "acquire", "acquires", "acquired", "deal", "deals",
    "launch", "launches", "launching", "launched", "revenue", "revenues",
    "dividend", "dividends", "confidence", "confident", "optimism", "optimistic",
    # Recovery & Momentum
    "recovery", "recovering", "recovered", "rebound", "rebounds", "rebounding",
    "momentum", "upside", "upbeat", "bullish", "positive", "positives",
    "outpace", "outpaces", "outpacing", "top", "tops", "topping", "topped",
    # Financial terms
    "investment", "invest", "invests", "investing", "investor", "investors",
    "shareholder", "shareholders", "margin", "margins", "return", "returns",
    "success", "successful", "succeed", "succeeds", "win", "wins", "winning", "won",
    "best", "better", "improve", "improves", "improving", "improved", "improvement",
    # Additional bullish signals
    "boom", "thriving", "thrive", "thrives", "flourish", "flourishes",
    "dominate", "dominates", "dominating", "leader", "leading", "leads",
    "demand", "demands", "popular", "popularity", "hit", "hits",
    "milestone", "milestones", "achievement", "achievements", "achieve",
    "target", "targets", "beat", "beats", "crushing", "crush", "crushes",
    "exceeds", "blowout", "outstanding", "excellent", "exceptional",
    "remarkable", "incredible", "amazing", "fantastic", "tremendous",
    "massive", "huge", "significant", "substantially", "dramatically",
}

BEARISH_KEYWORDS = {
    # Decline indicators
    "plunge", "plunges", "plunging", "plunged", "drop", "drops", "dropping", "dropped",
    "fall", "falls", "falling", "fell", "decline", "declines", "declining", "declined",
    "tumble", "tumbles", "tumbling", "tumbled", "plummet", "plummets", "plummeting",
    "sink", "sinks", "sinking", "sank", "sunk", "crash", "crashes", "crashing", "crashed",
    "collapse", "collapses", "collapsing", "collapsed", "dive", "dives", "diving",
    "slump", "slumps", "slumping", "slumped", "slide", "slides", "sliding", "slid",
    # Negative performance
    "miss", "misses", "missing", "missed", "disappoint", "disappoints", "disappointing",
    "disappointed", "disappointment", "underperform", "underperforms", "underperformed",
    "weak", "weaker", "weakest", "weakness", "poor", "poorer", "poorest",
    "loss", "losses", "losing", "lost", "lose", "loses", "negative", "negatives",
    # Business problems
    "layoff", "layoffs", "layoffs", "cut", "cuts", "cutting", "slash", "slashes",
    "slashing", "slashed", "downgrade", "downgrades", "downgrading", "downgraded",
    "warning", "warns", "warned", "warn", "caution", "cautious", "cautioning",
    "concern", "concerns", "concerned", "concerning", "worry", "worries", "worried",
    "threat", "threatens", "threatening", "threatened", "risk", "risks", "risky",
    # Legal & regulatory
    "lawsuit", "lawsuits", "sue", "sues", "suing", "sued", "investigation",
    "investigations", "investigate", "investigated", "probe", "probes", "probing",
    "fraud", "fraudulent", "scandal", "scandals", "violation", "violations",
    "penalty", "penalties", "fine", "fines", "fined", "regulatory", "regulation",
    # Crisis terms
    "crisis", "crises", "emergency", "bankruptcy", "bankrupt", "default", "defaults",
    "defaulting", "defaulted", "recession", "recessions", "downturn", "downturns",
    "trouble", "troubles", "troubled", "problem", "problems", "issue", "issues",
    # Selling & bearish
    "sell", "sells", "selling", "sold", "selloff", "sell-off", "bearish",
    "pessimistic", "pessimism", "downside", "fear", "fears", "feared", "fearing",
    # Additional bearish signals
    "shortfall", "breach", "breaches", "breached", "dispute", "disputes",
    "stall", "stalls", "stalling", "stalled", "halt", "halts", "halting", "halted",
    "suspend", "suspends", "suspending", "suspended", "delay", "delays", "delayed",
    "struggle", "struggles", "struggling", "struggled", "fail", "fails", "failing",
    "failed", "failure", "failures", "worst", "worse", "worsening", "worsened",
    "decline", "shrink", "shrinks", "shrinking", "shrank", "shrunk",
    "pressure", "pressures", "pressured", "headwind", "headwinds",
    "setback", "setbacks", "obstacle", "obstacles", "challenge", "challenges",
    "turmoil", "chaos", "volatile", "volatility", "uncertainty", "uncertain",
    "recall", "recalls", "recalled", "defect", "defects", "defective",
    "terminate", "terminates", "terminated", "termination", "exit", "exits",
}

NEUTRAL_KEYWORDS = {
    # Stability indicators
    "unchanged", "flat", "steady", "stable", "stability", "stabilize", "stabilizes",
    "sideways", "nowhere", "range", "rangebound", "range-bound", "consolidate",
    "consolidates", "consolidating", "consolidated", "consolidation",
    # Neutral actions
    "maintain", "maintains", "maintained", "maintaining", "hold", "holds",
    "holding", "held", "remain", "remains", "remaining", "remained",
    "stay", "stays", "staying", "stayed", "keep", "keeps", "keeping", "kept",
    # Expected/Normal
    "expected", "expects", "expecting", "inline", "in-line", "line", "target",
    "normal", "typical", "usual", "average", "moderate", "modest", "modestly",
    # Mixed signals
    "mixed", "neutral", "balance", "balanced", "balances", "even", "evenly",
    "muted", "subdued", "calm", "quiet", "little", "slight", "slightly",
    # No change
    "no change", "unchanged", "unaffected", "steady-state", "status quo",
    "continues", "continue", "continuing", "continued", "ongoing",
    "wait", "waits", "waiting", "waited", "pause", "pauses", "pausing", "paused",
    # Market terms (neutral)
    "trading", "traded", "trades", "volume", "session", "market", "markets",
    "sector", "sectors", "industry", "industries", "quarterly", "annual",
    # Additional neutral signals
    "report", "reports", "reported", "reporting", "announce", "announces",
    "announced", "announcing", "meeting", "meetings", "conference",
    "guidance", "forecast", "forecasts", "outlook", "estimate", "estimates",
    "review", "reviews", "reviewed", "reviewing", "assess", "assesses",
    "assessed", "assessing", "evaluation", "evaluate", "evaluates",
    "restructure", "restructures", "restructured", "restructuring",
    "transition", "transitions", "transitioning", "transitioned",
    "update", "updates", "updated", "updating", "release", "releases",
}



def compute_vader_sentiment(texts: pd.Series) -> pd.DataFrame:
    """Compute VADER sentiment scores (optimized for social media / news)."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        
        def _get_vader(text):
            scores = analyzer.polarity_scores(str(text) if text else "")
            return scores
        
        vader_results = texts.fillna("").apply(_get_vader)
        return pd.DataFrame({
            "vader_compound": vader_results.apply(lambda x: x["compound"]),
            "vader_positive": vader_results.apply(lambda x: x["pos"]),
            "vader_negative": vader_results.apply(lambda x: x["neg"]),
            "vader_neutral": vader_results.apply(lambda x: x["neu"]),
        })
    except ImportError:
        logger.warning("vaderSentiment not installed, returning zeros")
        n = len(texts)
        return pd.DataFrame({
            "vader_compound": [0.0] * n,
            "vader_positive": [0.0] * n,
            "vader_negative": [0.0] * n,
            "vader_neutral": [0.0] * n,
        })


def compute_textblob_sentiment(texts: pd.Series) -> pd.DataFrame:
    """Compute polarity, subjectivity, keyword signals, and VADER scores."""
    results = texts.fillna("").apply(lambda t: TextBlob(t).sentiment)
    polarity = results.apply(lambda s: s.polarity)
    subjectivity = results.apply(lambda s: s.subjectivity)

    # Keyword-based financial sentiment features
    # Match against both individual words AND bigrams for compound terms
    def _keyword_feats(text: str):
        text_lower = text.lower()
        words = set(text_lower.split())
        bull = len(words & BULLISH_KEYWORDS)
        bear = len(words & BEARISH_KEYWORDS)
        neut = len(words & NEUTRAL_KEYWORDS)
        # Also check for any keyword substring (catches lemmatized forms)
        for kw in BULLISH_KEYWORDS:
            if kw in text_lower and kw not in words:
                bull += 1
        for kw in BEARISH_KEYWORDS:
            if kw in text_lower and kw not in words:
                bear += 1
        for kw in NEUTRAL_KEYWORDS:
            if kw in text_lower and kw not in words:
                neut += 1
        return bull, bear, neut

    kw = texts.fillna("").apply(_keyword_feats)
    bullish_count = kw.apply(lambda x: x[0])
    bearish_count = kw.apply(lambda x: x[1])
    neutral_count = kw.apply(lambda x: x[2])

    # VADER sentiment scores
    vader_df = compute_vader_sentiment(texts)

    return pd.DataFrame(
        {
            "sentiment_polarity": polarity,
            "sentiment_subjectivity": subjectivity,
            "sentiment_intensity": polarity.abs(),
            "polarity_x_subjectivity": polarity * subjectivity,
            "bullish_keyword_count": bullish_count,
            "bearish_keyword_count": bearish_count,
            "neutral_keyword_count": neutral_count,
            "keyword_sentiment": bullish_count - bearish_count,
            "neutral_signal": (neutral_count > 0).astype(int),
            # VADER scores
            "vader_compound": vader_df["vader_compound"].values,
            "vader_positive": vader_df["vader_positive"].values,
            "vader_negative": vader_df["vader_negative"].values,
            "vader_neutral": vader_df["vader_neutral"].values,
        }
    )


def compute_finbert_sentiment(texts: list) -> pd.DataFrame:
    """Financial sentiment classification via FinBERT.

    Returns a DataFrame with columns: ``fin_sentiment_label``,
    ``fin_sentiment_score``.
    """
    try:
        from transformers import pipeline

        clf = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_MODEL,
            tokenizer=SENTIMENT_MODEL,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )
        results = clf(texts)
        return pd.DataFrame(
            {
                "fin_sentiment_label": [r["label"] for r in results],
                "fin_sentiment_score": [r["score"] for r in results],
            }
        )
    except Exception as exc:
        logger.warning("FinBERT sentiment unavailable: %s", exc)
        return pd.DataFrame(
            {
                "fin_sentiment_label": ["neutral"] * len(texts),
                "fin_sentiment_score": [0.0] * len(texts),
            }
        )


# ═══════════════════════════════════════════════════════════════════
# 3. MARKET FEATURES
# ═══════════════════════════════════════════════════════════════════
def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive technical indicators from OHLCV data.

    Expects columns: ``Close``, ``Volume`` (and ideally ``High``, ``Low``).
    """
    out = pd.DataFrame(index=df.index)

    close = df["Close"] if "Close" in df.columns else df.filter(like="close").iloc[:, 0]
    volume = df["Volume"] if "Volume" in df.columns else df.filter(like="volume").iloc[:, 0]

    # Moving averages
    for w in MOVING_AVERAGE_WINDOWS:
        out[f"ma_{w}"] = close.rolling(w, min_periods=1).mean()
        out[f"ma_ratio_{w}"] = close / out[f"ma_{w}"]

    # Volatility (rolling std of returns)
    returns = close.pct_change()
    out["volatility"] = returns.rolling(VOLATILITY_WINDOW, min_periods=1).std()

    # Momentum (rate-of-change)
    out["momentum"] = close.pct_change(MOMENTUM_WINDOW)

    # Volume features
    out["volume_ma"] = volume.rolling(20, min_periods=1).mean()
    out["volume_ratio"] = volume / out["volume_ma"].replace(0, np.nan)

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    return out.fillna(0)


# ═══════════════════════════════════════════════════════════════════
# 4. TEMPORAL FEATURES
# ═══════════════════════════════════════════════════════════════════
def compute_temporal_features(dates: pd.Series) -> pd.DataFrame:
    """Extract time-based signals from publication timestamps."""
    dates = pd.to_datetime(dates, errors="coerce")
    out = pd.DataFrame(index=dates.index)
    out["hour"] = dates.dt.hour
    out["day_of_week"] = dates.dt.dayofweek
    out["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)

    # Market session (US Eastern approximate)
    out["market_session"] = 0  # pre-market
    out.loc[(dates.dt.hour >= 9) & (dates.dt.hour < 16), "market_session"] = 1
    out.loc[dates.dt.hour >= 16, "market_session"] = 2  # after-hours

    # Cyclical encoding of hour & day
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    return out.fillna(0)


# ═══════════════════════════════════════════════════════════════════
# COMBINED FEATURE MATRIX
# ═══════════════════════════════════════════════════════════════════
def build_feature_matrix(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    date_col: str = "published_at",
    use_finbert: bool = False,
    use_sentence_embeddings: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the complete feature matrix and label vector.

    Returns
    -------
    X : np.ndarray  (n_samples, n_features)
    y : np.ndarray  (n_samples,)
    """
    parts = []

    # Text features (TF-IDF)
    tfidf_ext = TextFeatureExtractor()
    tfidf_feats = tfidf_ext.fit_transform(df[text_col])
    parts.append(tfidf_feats)
    logger.info("TF-IDF features: %s", tfidf_feats.shape)

    # Sentence embeddings (MiniLM - fast, 384-dim)
    if use_sentence_embeddings:
        sent_emb = get_sentence_embeddings(df[text_col].tolist())
        parts.append(sent_emb)
        logger.info("Sentence embeddings: %s", sent_emb.shape)

    # Sentiment features (TextBlob + VADER + keywords)
    sentiment = compute_textblob_sentiment(df[text_col])
    parts.append(sentiment.values)
    logger.info("Sentiment features: %s", sentiment.shape)

    # FinBERT embeddings (optional — heavy)
    if use_finbert:
        emb = get_finbert_embeddings(df[text_col].tolist())
        parts.append(emb)
        logger.info("FinBERT embeddings: %s", emb.shape)

    # Market features  (only if price data is present)
    if "Close" in df.columns or any("close" in c.lower() for c in df.columns):
        market = compute_market_features(df)
        parts.append(market.values)
        logger.info("Market features: %s", market.shape)

    # Temporal features
    if date_col in df.columns:
        temporal = compute_temporal_features(df[date_col])
        parts.append(temporal.values)
        logger.info("Temporal features: %s", temporal.shape)

    X = np.hstack(parts)
    y = df["label"].values.astype(int)

    logger.info("Final feature matrix: X=%s, y=%s", X.shape, y.shape)
    return X, y
