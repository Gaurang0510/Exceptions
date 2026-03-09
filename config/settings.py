"""
News2TradeAI — Central Configuration
=====================================
All project-wide constants, API keys, model hyper-parameters, and path
definitions live here so every module draws from a single source of truth.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT_DIR / "models" / "saved"
LOG_DIR = ROOT_DIR / "logs"
CACHE_DIR = ROOT_DIR / "cache"

for _d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─── API Keys (loaded from .env) ─────────────────────────────────────────────
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# ─── Label Generation ────────────────────────────────────────────────────────
PREDICTION_HORIZON_HOURS = 24          # predict movement within this window
UP_THRESHOLD = 0.01                    # return > +1 % → UP
DOWN_THRESHOLD = -0.01                 # return < −1 % → DOWN

# ─── Model Hyper-parameters ──────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_SEQUENCE_LENGTH = 128              # for transformer / LSTM inputs
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 10

# ─── Feature Engineering ─────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
NGRAM_RANGE = (1, 3)
SENTIMENT_MODEL = "ProsusAI/finbert"   # HuggingFace model id

# ─── Market Indicators ───────────────────────────────────────────────────────
MOVING_AVERAGE_WINDOWS = [5, 10, 20, 50]
VOLATILITY_WINDOW = 20
MOMENTUM_WINDOW = 14

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501

# ─── API Server ───────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ─── Default Tickers ─────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA"]

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s"
