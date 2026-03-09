"""
News2Trade AI - Sentiment Analysis Engine
Uses FinBERT (financial-domain BERT) + VADER as a lightweight fallback.
"""
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Try loading FinBERT (transformers + torch)
_FINBERT_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    _FINBERT_AVAILABLE = True
except ImportError:
    pass


class SentimentAnalyzer:
    """
    Dual-mode sentiment analyzer:
      • Primary:  FinBERT (ProsusAI/finbert) – fine-tuned on financial text
      • Fallback: VADER – rule-based, fast, no GPU needed
    """

    def __init__(self, use_finbert: bool = True):
        self.vader = SentimentIntensityAnalyzer()
        self.finbert_pipeline = None

        if use_finbert and _FINBERT_AVAILABLE:
            try:
                print("[INFO] Loading FinBERT model... (first run downloads ~400 MB)")
                self.finbert_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    top_k=None,           # return all labels with scores
                    truncation=True,
                    max_length=512,
                )
                print("[INFO] FinBERT loaded successfully.")
            except Exception as e:
                print(f"[WARN] Could not load FinBERT: {e}. Falling back to VADER.")
                self.finbert_pipeline = None

    # ─── Public API ────────────────────────────────────────────────

    def analyze(self, text: str) -> dict:
        """
        Analyze a single text string.
        Returns:
            {
                "sentiment_score": float [-1, 1],
                "sentiment_label": str (positive / negative / neutral),
                "confidence": float [0, 1],
                "method": str (finbert / vader),
                "details": dict
            }
        """
        if not text or not text.strip():
            return self._empty_result()

        if self.finbert_pipeline is not None:
            return self._analyze_finbert(text)
        return self._analyze_vader(text)

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        """Analyze a list of texts."""
        return [self.analyze(t) for t in texts]

    # ─── FinBERT Analysis ──────────────────────────────────────────

    def _analyze_finbert(self, text: str) -> dict:
        try:
            results = self.finbert_pipeline(text[:512])  # truncate
            if isinstance(results, list) and isinstance(results[0], list):
                results = results[0]

            scores = {r["label"]: r["score"] for r in results}
            pos = scores.get("positive", 0)
            neg = scores.get("negative", 0)
            neu = scores.get("neutral", 0)

            # Composite score: [-1, 1]
            sentiment_score = pos - neg

            # Determine label
            best_label = max(scores, key=scores.get)
            confidence = scores[best_label]

            return {
                "sentiment_score": round(sentiment_score, 4),
                "sentiment_label": best_label,
                "confidence": round(confidence, 4),
                "method": "finbert",
                "details": {
                    "positive": round(pos, 4),
                    "negative": round(neg, 4),
                    "neutral": round(neu, 4),
                },
            }
        except Exception as e:
            print(f"[WARN] FinBERT error: {e}. Falling back to VADER.")
            return self._analyze_vader(text)

    # ─── VADER Analysis ────────────────────────────────────────────

    def _analyze_vader(self, text: str) -> dict:
        scores = self.vader.polarity_scores(text)
        compound = scores["compound"]  # already in [-1, 1]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        # Map compound to confidence-like metric
        confidence = abs(compound)

        return {
            "sentiment_score": round(compound, 4),
            "sentiment_label": label,
            "confidence": round(confidence, 4),
            "method": "vader",
            "details": {
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral": round(scores["neu"], 4),
                "compound": round(compound, 4),
            },
        }

    # ─── Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> dict:
        return {
            "sentiment_score": 0.0,
            "sentiment_label": "neutral",
            "confidence": 0.0,
            "method": "none",
            "details": {},
        }

    @staticmethod
    def score_to_category(score: float) -> str:
        """Convert numeric score to human-readable category."""
        if score >= 0.6:
            return "Very Positive"
        elif score >= 0.2:
            return "Positive"
        elif score >= -0.2:
            return "Neutral"
        elif score >= -0.6:
            return "Negative"
        else:
            return "Very Negative"
