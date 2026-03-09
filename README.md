# News2Trade AI 📈
### Explainable, Sentiment-Driven Trading Signals

---

## 🧠 Problem Statement

Financial markets react instantly to news, but retail users struggle to:
- **Understand** the sentiment behind financial news
- **Judge** whether news is over-hyped or misleading
- **Decide** Buy, Sell, or Hold without expert knowledge

This often leads to **emotional and risky decisions**.

---

## 💡 Solution

**News2Trade AI** converts financial news into **actionable, explainable trading signals** using NLP and Machine Learning.

| Capability | Description |
|---|---|
| 📰 Sentiment Analysis | FinBERT + VADER dual-engine NLP |
| ⚠️ Hype / Fake-News Detection | Multi-signal credibility scoring |
| 🎯 Buy / Sell / Hold Prediction | Random Forest ML classifier |
| 🛡️ Beginner Safety Mode | Filters risky low-confidence signals |
| 📊 Explainability | Feature importances, confidence scores, reasoning |

---

## 🏗️ Architecture

```
User Input (news headline / article)
         │
         ▼
┌──────────────────────┐
│   News API Fetcher   │ ← NewsAPI.org (or built-in sample dataset)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Sentiment Analyzer  │ ← FinBERT (financial BERT) + VADER fallback
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    Hype Detector     │ ← Keyword + punctuation + subjectivity + source check
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Trading Signal ML   │ ← Random Forest on 12 engineered features
│  (Buy / Sell / Hold) │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Beginner Safety     │ ← Overrides low-confidence signals to HOLD
│  Mode Filter         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Streamlit Dashboard │ ← Interactive UI with charts & explanations
└──────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
cd News2TradeAI
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. (Optional) Add News API Key
Get a free key from [newsapi.org](https://newsapi.org/register), then:
```bash
copy .env.example .env
# Edit .env and paste your key
```
> **Note:** The app works without an API key using a built-in sample dataset of 28 financial news articles.

### 3. Run the App
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`

---

## 📁 Project Structure

```
News2TradeAI/
├── app.py                 # Streamlit web UI
├── pipeline.py            # End-to-end orchestrator
├── news_fetcher.py        # NewsAPI data fetcher + sample dataset
├── sentiment_analyzer.py  # FinBERT + VADER sentiment engine
├── hype_detector.py       # Hype / fake-news detector
├── trading_model.py       # Random Forest Buy/Sell/Hold classifier
├── config.py              # All configuration & thresholds
├── requirements.txt       # Python dependencies
├── .env.example           # API key template
├── models/                # Saved ML models (auto-created)
└── README.md              # This file
```

---

## 🔬 Technical Details

### Sentiment Analysis (Dual Engine)
| Engine | Type | Best For |
|---|---|---|
| **FinBERT** | Transformer (ProsusAI/finbert) | Accurate financial sentiment |
| **VADER** | Rule-based lexicon | Fast fallback, no GPU needed |

### Hype Detection (4-Signal Scoring)
| Signal | Weight | Method |
|---|---|---|
| Keyword matching | 40% | Clickbait phrase detection |
| Subjectivity | 25% | TextBlob sentiment subjectivity |
| Punctuation abuse | 20% | Excessive !, ?, CAPS ratio |
| Source credibility | 15% | Trusted vs unknown sources |

### ML Model (Random Forest)
- **12 engineered features** from sentiment + hype analysis
- **200 trees**, max depth 10, balanced class weights
- Trained on analysed news data with synthetic labels
- Cross-validated with accuracy reporting
- Feature importance for explainability

### Beginner Safety Mode 🛡️
| Setting | Standard | Beginner |
|---|---|---|
| Confidence threshold | 60% | 75% |
| Hype override | Yes | Yes (stricter) |
| Low-confidence signals | Shown | Forced to HOLD |

---

## 📊 Features at a Glance

- **News Feed Analysis** — Batch-analyze 10-50 articles at once
- **Custom Article Input** — Paste any headline for instant signal
- **Interactive Dashboard** — Pie charts, scatter plots, histograms
- **Deep Dive View** — Click any article for full explanation
- **CSV Export** — Download results for further analysis
- **Hype Gauge** — Visual hype risk meter per article
- **Probability Breakdown** — Buy/Sell/Hold probability bars

---

## ⚖️ Disclaimer

> **This tool is for educational and research purposes only.**
> It does not constitute financial advice. Always consult a qualified
> financial advisor before making investment decisions. Past performance
> and model predictions do not guarantee future results.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend | Streamlit |
| NLP | FinBERT, VADER, TextBlob |
| ML | scikit-learn (Random Forest) |
| Data API | NewsAPI.org |
| Visualization | Plotly |
| Explainability | Feature importances, SHAP-ready |
| Language | Python 3.10+ |

---

## 📜 License

MIT License — free for educational use.
