"""
Real-Time Prediction Pipeline
================================
Accepts a financial news headline, preprocesses it, extracts features,
and returns a prediction with confidence scores and explanations.
"""

import logging
from typing import Dict, Any, Optional
from collections import Counter

import numpy as np
import joblib

from config.settings import MODEL_DIR
from preprocessing.text_pipeline import (
    preprocess_text,
    extract_financial_keywords,
    extract_entities,
)
from preprocessing.feature_engineering import (
    TextFeatureExtractor,
    compute_textblob_sentiment,
    get_single_embedding,
    BULLISH_KEYWORDS,
    BEARISH_KEYWORDS,
    NEUTRAL_KEYWORDS,
)
from data.dataset_builder import LABEL_MAP

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """End-to-end inference pipeline for news → stock-impact prediction.

    Usage
    -----
    >>> pipeline = PredictionPipeline("xgboost")
    >>> result = pipeline.predict("Apple reports record quarterly earnings")
    >>> print(result["prediction"], result["confidence"])
    """

    def __init__(self, model_name: str = "xgboost"):
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        path = MODEL_DIR / f"{self.model_name}.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No saved model at {path}. Run the training pipeline first."
            )
        checkpoint = joblib.load(path)
        self.model = checkpoint["model"]
        self.scaler = checkpoint.get("scaler")
        # Also load the TF-IDF vectorizer if saved separately
        vec_path = MODEL_DIR / "tfidf_vectorizer.joblib"
        self.vectorizer = joblib.load(vec_path) if vec_path.exists() else None
        logger.info("Loaded model: %s", self.model_name)

    def predict(self, headline: str) -> Dict[str, Any]:
        """Run the full prediction on a single headline string."""
        # 1. Preprocess
        clean = preprocess_text(headline)

        # 2. TF-IDF features
        if self.vectorizer is not None:
            tfidf = self.vectorizer.transform([clean]).toarray()
        else:
            tfidf = np.zeros((1, 100))
            logger.warning("No TF-IDF vectorizer found — using zero features.")

        # 3. Sentence embeddings (MiniLM, 384-dim)
        sent_emb = get_single_embedding(clean)

        # 4. Sentiment — must match training features exactly (13 features)
        from textblob import TextBlob

        blob = TextBlob(clean)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Use BOTH raw headline and clean text for keyword matching
        # (training data sees clean text; raw text catches more inflections)
        raw_lower = headline.lower()
        clean_lower = clean.lower()
        combined = raw_lower + " " + clean_lower
        words = set(combined.split())
        bull = len(words & BULLISH_KEYWORDS)
        bear = len(words & BEARISH_KEYWORDS)
        neut = len(words & NEUTRAL_KEYWORDS)
        # Also check substring matches for multi-word / partial forms
        for kw in BULLISH_KEYWORDS:
            if kw in combined and kw not in words:
                bull += 1
        for kw in BEARISH_KEYWORDS:
            if kw in combined and kw not in words:
                bear += 1
        for kw in NEUTRAL_KEYWORDS:
            if kw in combined and kw not in words:
                neut += 1

        # VADER sentiment
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vader = SentimentIntensityAnalyzer()
            vs = vader.polarity_scores(headline)
            vader_compound = vs["compound"]
            vader_pos = vs["pos"]
            vader_neg = vs["neg"]
            vader_neu = vs["neu"]
        except ImportError:
            vader_compound = vader_pos = vader_neg = vader_neu = 0.0

        sentiment_feats = np.array([[
            polarity,
            subjectivity,
            abs(polarity),               # sentiment_intensity
            polarity * subjectivity,     # polarity_x_subjectivity
            bull,                         # bullish_keyword_count
            bear,                         # bearish_keyword_count
            neut,                         # neutral_keyword_count
            bull - bear,                  # keyword_sentiment
            1 if neut > 0 else 0,         # neutral_signal
            vader_compound,               # vader_compound
            vader_pos,                    # vader_positive
            vader_neg,                    # vader_negative
            vader_neu,                    # vader_neutral
        ]])

        # 5. Combine — order must match training: TF-IDF, sentence emb, sentiment
        #    For market & temporal features that exist in training but are
        #    unavailable at inference, pad with the scaler's mean.
        X = np.hstack([tfidf, sent_emb, sentiment_feats])
        if self.scaler is not None:
            expected = self.scaler.n_features_in_
            if X.shape[1] < expected:
                # Use scaler mean for missing features → they scale to 0
                n_missing = expected - X.shape[1]
                pad = self.scaler.mean_[-n_missing:].reshape(1, -1)
                X = np.hstack([X, pad])

        # 6. Scale
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # 7. Predict
        pred_class = int(self.model.predict(X)[0])
        proba = (
            self.model.predict_proba(X)[0].tolist()
            if hasattr(self.model, "predict_proba")
            else [0.0, 0.0, 0.0]
        )

        # 8. Enrichment
        keywords = extract_financial_keywords(headline)
        entities = extract_entities(headline)

        result = {
            "headline": headline,
            "clean_text": clean,
            "prediction": LABEL_MAP[pred_class],
            "prediction_id": pred_class,
            "confidence": max(proba),
            "probabilities": {
                "DOWN": round(proba[0], 4),
                "NEUTRAL": round(proba[1], 4),
                "UP": round(proba[2], 4),
            },
            "sentiment": {
                "polarity": round(polarity, 4),
                "subjectivity": round(subjectivity, 4),
                "vader_compound": round(vader_compound, 4),
                "vader_positive": round(vader_pos, 4),
                "vader_negative": round(vader_neg, 4),
            },
            "financial_keywords": keywords,
            "entities": entities,
            "model_used": self.model_name,
        }
        return result

    def predict_batch(self, headlines: list) -> list:
        """Run predictions on a list of headlines."""
        return [self.predict(h) for h in headlines]


# ────────────────────────────────────────────────────────────────────
# Ensemble Pipeline - Uses all models for robust predictions
# ────────────────────────────────────────────────────────────────────
class EnsemblePipeline:
    """Ensemble pipeline that aggregates predictions from all available models.
    
    Usage
    -----
    >>> pipeline = EnsemblePipeline(beginner_mode=True)
    >>> result = pipeline.predict("Apple reports record earnings")
    """
    
    AVAILABLE_MODELS = ["random_forest", "xgboost", "svm", "logistic_regression"]
    
    def __init__(self, beginner_mode: bool = False):
        self.beginner_mode = beginner_mode
        self.pipelines = {}
        self._load_models()
        
        # Beginner mode thresholds
        self.min_confidence = 0.70 if beginner_mode else 0.50
        self.min_agreement = 0.60 if beginner_mode else 0.50  # % of models must agree
        
    def _load_models(self):
        """Load all available models."""
        for model_name in self.AVAILABLE_MODELS:
            try:
                self.pipelines[model_name] = PredictionPipeline(model_name)
            except FileNotFoundError:
                logger.warning(f"Model {model_name} not found, skipping.")
                
        if not self.pipelines:
            raise FileNotFoundError("No trained models found. Run training first.")
            
    def predict(self, headline: str) -> Dict[str, Any]:
        """Run prediction using all models and aggregate results."""
        results = {}
        all_probs = {"DOWN": [], "NEUTRAL": [], "UP": []}
        predictions = []
        
        # Get prediction from each model
        for name, pipeline in self.pipelines.items():
            try:
                result = pipeline.predict(headline)
                results[name] = result
                predictions.append(result["prediction"])
                for label in all_probs:
                    all_probs[label].append(result["probabilities"][label])
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                
        if not results:
            raise RuntimeError("All models failed to make predictions.")
            
        # Aggregate probabilities (mean across models)
        avg_probs = {
            label: float(np.mean(probs)) for label, probs in all_probs.items()
        }
        
        # Count votes for each prediction
        votes = Counter(predictions)
        total_models = len(predictions)
        
        # Determine ensemble prediction
        raw_prediction = max(avg_probs, key=avg_probs.get)
        raw_confidence = avg_probs[raw_prediction]
        
        # Calculate agreement ratio
        agreement = votes[raw_prediction] / total_models if total_models > 0 else 0
        
        # Apply beginner mode safety rules
        final_prediction = raw_prediction
        safety_applied = False
        safety_reason = None
        
        if self.beginner_mode:
            # Rule 1: Low confidence → HOLD
            if raw_confidence < self.min_confidence:
                final_prediction = "NEUTRAL"
                safety_applied = True
                safety_reason = f"Confidence too low ({raw_confidence:.1%} < {self.min_confidence:.0%})"
                
            # Rule 2: Models disagree → HOLD
            elif agreement < self.min_agreement:
                final_prediction = "NEUTRAL"
                safety_applied = True
                safety_reason = f"Models disagree ({agreement:.0%} < {self.min_agreement:.0%} agreement)"
                
            # Rule 3: Close call between UP and DOWN → HOLD
            elif raw_prediction in ["UP", "DOWN"]:
                opposite = "DOWN" if raw_prediction == "UP" else "UP"
                if abs(avg_probs[raw_prediction] - avg_probs[opposite]) < 0.15:
                    final_prediction = "NEUTRAL"
                    safety_applied = True
                    safety_reason = "UP/DOWN too close - uncertain market direction"
        
        # Build action recommendation
        if self.beginner_mode:
            action_map = {
                "UP": "🟢 CONSIDER BUYING - But start small, use stop-loss",
                "DOWN": "🔴 CONSIDER SELLING - Or wait for better entry",
                "NEUTRAL": "🟡 HOLD/WAIT - Not a clear signal right now",
            }
        else:
            action_map = {
                "UP": "📈 Bullish signal",
                "DOWN": "📉 Bearish signal", 
                "NEUTRAL": "➡️ Neutral/Hold",
            }
            
        # Use first model's result for sentiment/keywords (they're all the same)
        first_result = next(iter(results.values()))
        
        return {
            "headline": headline,
            "clean_text": first_result["clean_text"],
            "prediction": final_prediction,
            "raw_prediction": raw_prediction,
            "confidence": raw_confidence,
            "probabilities": {k: round(v, 4) for k, v in avg_probs.items()},
            "model_votes": dict(votes),
            "agreement": round(agreement, 2),
            "models_used": list(results.keys()),
            "individual_predictions": {k: v["prediction"] for k, v in results.items()},
            "sentiment": first_result["sentiment"],
            "financial_keywords": first_result["financial_keywords"],
            "entities": first_result.get("entities", []),
            "beginner_mode": self.beginner_mode,
            "safety_applied": safety_applied,
            "safety_reason": safety_reason,
            "action": action_map[final_prediction],
        }


# ────────────────────────────────────────────────────────────────────
# Convenience singleton
# ────────────────────────────────────────────────────────────────────
_PIPELINE: Optional[PredictionPipeline] = None
_ENSEMBLE: Optional[EnsemblePipeline] = None


def get_pipeline(model_name: str = "xgboost") -> PredictionPipeline:
    global _PIPELINE
    if _PIPELINE is None or _PIPELINE.model_name != model_name:
        _PIPELINE = PredictionPipeline(model_name)
    return _PIPELINE


def get_ensemble(beginner_mode: bool = False) -> EnsemblePipeline:
    global _ENSEMBLE
    if _ENSEMBLE is None or _ENSEMBLE.beginner_mode != beginner_mode:
        _ENSEMBLE = EnsemblePipeline(beginner_mode=beginner_mode)
    return _ENSEMBLE
