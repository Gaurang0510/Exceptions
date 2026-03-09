"""
News2Trade AI - Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ─── API Configuration ───────────────────────────────────────────────
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# ─── Model Configuration ─────────────────────────────────────────────
MODEL_PATH = "models/"
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"  # Financial BERT for sentiment

# ─── Signal Thresholds ───────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.60          # Minimum confidence for any signal
BEGINNER_CONFIDENCE_THRESHOLD = 0.75 # Stricter threshold for beginners
HYPE_SCORE_THRESHOLD = 0.65         # Above this → flagged as potential hype

# ─── Sentiment Score Ranges ──────────────────────────────────────────
SENTIMENT_BINS = {
    "very_negative": (-1.0, -0.6),
    "negative":      (-0.6, -0.2),
    "neutral":       (-0.2,  0.2),
    "positive":      ( 0.2,  0.6),
    "very_positive": ( 0.6,  1.0),
}

# ─── Financial Keywords ──────────────────────────────────────────────
FINANCIAL_KEYWORDS = [
    "stock", "market", "shares", "earnings", "revenue", "profit", "loss",
    "dividend", "IPO", "merger", "acquisition", "bull", "bear", "rally",
    "crash", "inflation", "interest rate", "fed", "GDP", "recession",
    "bitcoin", "crypto", "forex", "bond", "yield", "nasdaq", "dow",
    "s&p", "nifty", "sensex", "trading", "investor", "portfolio",
]

# ─── Hype / Clickbait Indicators ─────────────────────────────────────
HYPE_INDICATORS = [
    "guaranteed", "100%", "get rich", "skyrocket", "moon", "to the moon",
    "massive gains", "never seen before", "once in a lifetime",
    "secret", "they don't want you to know", "insider", "urgent",
    "act now", "last chance", "breaking", "explosive", "triple",
    "quadruple", "10x", "100x", "free money", "no risk", "sure thing",
    "can't lose", "millionaire", "billionaire overnight",
]
