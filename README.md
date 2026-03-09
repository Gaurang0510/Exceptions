# 📈 News2TradeAI — Financial News Impact Predictor

> **Research-grade ML system** that predicts stock price movement (UP / DOWN / NEUTRAL) from financial news using Natural Language Processing, Machine Learning, and Explainable AI.

---

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [System Architecture](#2-system-architecture)
3. [Data Sources](#3-data-sources)
4. [Data Preprocessing Pipeline](#4-data-preprocessing-pipeline)
5. [Feature Engineering](#5-feature-engineering)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Training Pipeline](#7-training-pipeline)
8. [Model Evaluation](#8-model-evaluation)
9. [Explainable AI](#9-explainable-ai)
10. [Real-Time Prediction Pipeline](#10-real-time-prediction-pipeline)
11. [Interactive Dashboard](#11-interactive-dashboard)
12. [API Integration](#12-api-integration)
13. [Project Structure](#13-project-structure)
14. [Setup & Reproducibility](#14-setup--reproducibility)
15. [Quick Start](#15-quick-start)

---

## 1. Problem Formulation

### Task Definition

This is a **supervised multi-class classification** problem.

| Component | Description |
|-----------|-------------|
| **Input** | Financial news headline or article text |
| **Output** | Predicted stock movement category |
| **Classes** | `UP` (2), `NEUTRAL` (1), `DOWN` (0) |
| **Horizon** | Stock movement within **24 hours** after news publication |

### Label Generation

Labels are generated from **historical forward returns**:

```
forward_return = (price_t+1 / price_t) - 1

if forward_return >  +1%  →  UP
if forward_return <  -1%  →  DOWN
otherwise                 →  NEUTRAL
```

Thresholds are configurable in `config/settings.py`:

```python
UP_THRESHOLD   =  0.01   # +1%
DOWN_THRESHOLD = -0.01   # -1%
PREDICTION_HORIZON_HOURS = 24
```

### Why This Matters

Financial markets react to news in measurable ways. By combining NLP with market data, we can quantify the expected impact of news events — a key capability in quantitative finance research.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     NEWS2TRADEAI ARCHITECTURE                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│  │  Data Layer  │───▶│ Preprocessing│───▶│ Feature Engineering │  │
│  │             │    │   Pipeline   │    │                     │  │
│  │ • Yahoo API │    │ • Normalise  │    │ • TF-IDF / n-grams │  │
│  │ • NewsAPI   │    │ • Tokenise   │    │ • Sentiment scores │  │
│  │ • Finnhub   │    │ • Lemmatise  │    │ • Market indicators│  │
│  │ • Kaggle CSV│    │ • NER        │    │ • Temporal features│  │
│  └─────────────┘    └──────────────┘    └────────┬────────────┘  │
│                                                   │               │
│                                                   ▼               │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                   MODEL TRAINING PIPELINE                    │  │
│  │                                                              │  │
│  │  Logistic Reg ─ SVM ─ Random Forest ─ XGBoost ─ LSTM ─ BERT│  │
│  │                                                              │  │
│  │  • Cross-validation  • Hyper-parameter tuning  • Checkpoints│  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                              │                                    │
│              ┌───────────────┼───────────────┐                    │
│              ▼               ▼               ▼                    │
│  ┌───────────────┐  ┌──────────────┐  ┌─────────────────┐       │
│  │   Evaluation   │  │  Explainable │  │   Prediction    │       │
│  │   • Accuracy   │  │     AI       │  │    Pipeline     │       │
│  │   • F1 / AUC   │  │  • SHAP      │  │  • Real-time    │       │
│  │   • Confusion  │  │  • Attention  │  │  • Confidence   │       │
│  └───────────────┘  └──────────────┘  └───────┬─────────┘       │
│                                                │                  │
│              ┌─────────────────────────────────┼────────────┐     │
│              ▼                                 ▼            │     │
│  ┌───────────────────┐            ┌─────────────────────┐  │     │
│  │  FastAPI Server    │            │  Streamlit Dashboard │  │     │
│  │  /predict          │◀──────────▶│  • Candlestick      │  │     │
│  │  /predict/batch    │            │  • Sentiment gauge   │  │     │
│  │  /stock/{ticker}   │            │  • SHAP charts       │  │     │
│  └───────────────────┘            └─────────────────────┘  │     │
│                                                             │     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Sources

### Historical Data

| Source | Data Type | Access |
|--------|-----------|--------|
| **Yahoo Finance** (`yfinance`) | OHLCV stock prices | Free — no API key |
| **Kaggle datasets** | Financial news CSVs | Download to `data/raw/` |

### Real-Time APIs

| API | Purpose | Key Required |
|-----|---------|--------------|
| **Alpha Vantage** | Historical daily prices | Free tier key |
| **Finnhub** | Real-time quotes + company news | Free tier key |
| **NewsAPI** | Global financial news search | Free tier key |

### Dataset Construction

1. **Fetch** news headlines with timestamps  
2. **Fetch** stock prices for matching tickers  
3. **Align** by date — each headline maps to the stock's forward return  
4. **Label** using the threshold rules above  
5. **Persist** as a Parquet file for reproducibility

The `data/dataset_builder.py` module automates this entire pipeline.

---

## 4. Data Preprocessing Pipeline

Each stage is a modular function in `preprocessing/text_pipeline.py`:

| Stage | Function | Description |
|-------|----------|-------------|
| 1 | `normalise_text()` | Lowercase, strip unicode, collapse whitespace |
| 2 | `remove_punctuation()` | Strip all non-alphanumeric characters |
| 3 | `tokenize()` | NLTK word-level tokenisation |
| 4 | `remove_stopwords()` | English + financial domain stop-words |
| 5 | `lemmatize()` | WordNet lemmatisation for root form |
| 6 | `extract_financial_keywords()` | Match against 40+ financial terms |
| 7 | `extract_entities()` | spaCy NER — ORG, MONEY, PERCENT, GPE |

The composite `preprocess_text()` runs stages 1-5 in sequence. `preprocess_dataframe()` applies the full pipeline to a DataFrame.

---

## 5. Feature Engineering

### Feature Categories

| Category | Features | Rationale |
|----------|----------|-----------|
| **Text** | TF-IDF vectors, tri-grams, FinBERT embeddings (768-d) | Capture lexical and semantic content of news |
| **Sentiment** | TextBlob polarity/subjectivity, FinBERT financial sentiment | Quantify emotional tone & market relevance |
| **Market** | Moving averages (5/10/20/50d), volatility, momentum, RSI, volume ratio | Contextualise news with market state |
| **Temporal** | Hour, day-of-week, market session, cyclical encodings | News impact varies by timing |

### Why Multi-Modal Features?

Text alone tells you *what* the news says. Market features tell you *where* the stock already is. Combining them gives the model a richer signal — for example, positive earnings news has less upside impact when the stock already rallied +30% in the prior month.

---

## 6. Machine Learning Models

### Baseline Models

| Model | Strengths |
|-------|-----------|
| **Logistic Regression** | Fast, interpretable, strong TF-IDF baseline |
| **SVM (RBF)** | Effective in high-dimensional feature spaces |

### Tree-Based Ensembles

| Model | Strengths |
|-------|-----------|
| **Random Forest** | Handles non-linearity, robust to outliers, built-in feature importance |
| **XGBoost** | State-of-the-art tabular performance, regularisation, handles missing data |

### Deep Learning

| Model | Strengths |
|-------|-----------|
| **LSTM** | Captures temporal dependencies in price sequences |
| **FinBERT Classifier** | Pre-trained on financial text — transfer learning from domain corpus |

---

## 7. Training Pipeline

`training/train_pipeline.py` implements a professional ML workflow:

```python
# Full benchmark: train, tune, evaluate, compare all models
from training.train_pipeline import run_benchmark
summary = run_benchmark(X, y, tune=True)
```

| Step | Implementation |
|------|----------------|
| Train/test split | Stratified 80/20 split |
| Cross-validation | 5-fold stratified CV |
| Hyper-parameter tuning | `GridSearchCV` with F1-macro scoring |
| Feature scaling | `StandardScaler` (fit on train only) |
| Model checkpointing | `joblib` serialisation to `models/saved/` |

---

## 8. Model Evaluation

`training/evaluation.py` provides rigorous metrics:

| Metric | Purpose |
|--------|---------|
| **Accuracy** | Overall correctness |
| **Precision** | False-positive control (per-class & macro) |
| **Recall** | False-negative control |
| **F1 Score** | Harmonic mean of precision & recall |
| **ROC-AUC** | Discrimination ability (one-vs-rest) |
| **Confusion Matrix** | Per-class error analysis |

### Financial Interpretation

- **High precision on UP** → fewer false buy signals → lower trading loss  
- **High recall on DOWN** → catch more adverse events → better risk management  
- **ROC-AUC > 0.65** is meaningful in noisy financial data

---

## 9. Explainable AI

`training/explainability.py` provides three layers of interpretability:

### SHAP Values
Game-theoretic attributions showing each feature's contribution to every prediction. Supports both TreeExplainer (fast, for tree models) and KernelExplainer (model-agnostic fallback).

### Feature Importance
Built-in Mean Decrease in Impurity (MDI) from tree-based models — a quick global view.

### Attention Maps
Self-attention heatmaps from the FinBERT transformer — reveals which tokens the model focuses on when predicting stock impact.

### Why Interpretability Matters
In financial AI, model interpretability is not optional. Portfolio managers need to understand *why* a model flags a headline as impactful. Regulatory frameworks (MiFID II, SEC) increasingly require explainability for algorithmic trading decisions.

---

## 10. Real-Time Prediction Pipeline

`api/prediction_pipeline.py` provides an end-to-end inference pipeline:

```python
from api.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline("xgboost")
result = pipeline.predict("Apple reports record quarterly earnings")

# result contains:
# - prediction: "UP"
# - confidence: 0.73
# - probabilities: {"DOWN": 0.08, "NEUTRAL": 0.19, "UP": 0.73}
# - sentiment: {"polarity": 0.45, "subjectivity": 0.62}
# - financial_keywords: ["earnings"]
# - entities: [{"text": "Apple", "label": "ORG"}]
```

---

## 11. Interactive Dashboard

The Streamlit dashboard (`dashboard/app.py`) provides 8 interactive panels:

1. **News Input** — paste or select headlines
2. **Prediction Panel** — classification + confidence gauge
3. **Candlestick Chart** — live OHLCV with 20/50-day moving averages
4. **Sentiment Gauge** — polarity dial (-1 to +1)
5. **Sentiment Timeline** — polarity over multiple predictions
6. **Probability Distribution** — per-class bar chart
7. **SHAP Feature Importance** — interactive bar chart
8. **Prediction History** — tabular log of all session predictions

---

## 12. API Integration

### REST Endpoints (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health check |
| `/predict` | POST | Single headline prediction |
| `/predict/batch` | POST | Batch prediction |
| `/stock/{ticker}` | GET | Fetch historical stock data |

### Data API Modules

`data/stock_data.py` and `data/news_data.py` provide:

- **Rate limiting** — configurable delays between API calls
- **Disk caching** — Parquet-based with TTL
- **Error handling** — graceful fallback on API failures
- **Asynchronous support** — FastAPI endpoints are async-ready

---

## 13. Project Structure

```
News2TradeAI/
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Central configuration
│
├── data/
│   ├── __init__.py
│   ├── stock_data.py            # Stock price fetchers
│   ├── news_data.py             # News data fetchers
│   └── dataset_builder.py       # Label generation & merging
│
├── preprocessing/
│   ├── __init__.py
│   ├── text_pipeline.py         # NLP preprocessing stages
│   └── feature_engineering.py   # Feature extraction module
│
├── models/
│   ├── __init__.py
│   ├── classical_models.py      # LR, SVM, RF, XGBoost
│   ├── deep_models.py           # LSTM, FinBERT classifier
│   └── saved/                   # Serialised model checkpoints
│
├── training/
│   ├── __init__.py
│   ├── train_pipeline.py        # Training & benchmarking
│   ├── evaluation.py            # Metrics & visualisation
│   └── explainability.py        # SHAP, attention maps
│
├── api/
│   ├── __init__.py
│   ├── prediction_pipeline.py   # Inference pipeline
│   └── server.py                # FastAPI application
│
├── dashboard/
│   ├── __init__.py
│   └── app.py                   # Streamlit dashboard
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Logging configuration
│   └── helpers.py               # Common utilities
│
├── notebooks/                   # Jupyter exploration notebooks
├── logs/                        # Application logs
├── cache/                       # API response cache
│
├── app.py                       # CLI entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # API key template
├── .gitignore
└── README.md
```

---

## 14. Setup & Reproducibility

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd News2TradeAI

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model (for NER)
python -m spacy download en_core_web_sm

# 5. Download NLTK data (auto-downloaded on first run)
python -c "import nltk; nltk.download('all')"
```

### API Configuration

```bash
# Copy the template and add your keys
copy .env.example .env
# Edit .env with your API keys (optional — demo mode works without keys)
```

### Dataset

The system can operate in two modes:

1. **Demo mode** — auto-generates a synthetic dataset of 3,000 labeled samples
2. **Real data** — place a Kaggle financial news CSV in `data/raw/` and run the training pipeline

---

## 15. Quick Start

### Full Demo (Recommended)

```bash
python app.py demo
```

This will:
1. Generate a synthetic financial news dataset
2. Train and evaluate all ML models
3. Launch the interactive Streamlit dashboard

### Step-by-Step

```bash
# Train all models
python app.py train

# Predict a single headline
python app.py predict "Apple reports record quarterly earnings"

# Start the REST API
python app.py api

# Launch the dashboard
python app.py dashboard
```

### Dashboard Only (No Training Required)

```bash
streamlit run dashboard/app.py
```

The dashboard includes a heuristic fallback that works without trained models — useful for demonstration and exploration.

---

## Technical Highlights

| Capability | Implementation |
|-----------|---------------|
| **NLP Pipeline** | 7-stage modular pipeline (normalise → NER) |
| **Feature Engineering** | 4 feature categories, 50+ engineered features |
| **Model Zoo** | 6 models from Logistic Regression to FinBERT |
| **Hyper-parameter Tuning** | GridSearchCV with stratified CV |
| **Explainability** | SHAP values, feature importance, attention maps |
| **Real-Time Inference** | FastAPI server with sub-second latency |
| **Interactive Dashboard** | 8-panel Streamlit UI with live stock charts |
| **Reproducibility** | Seed control, checkpointing, Parquet persistence |

---

*Built for quantitative finance research and ML presentation.*
