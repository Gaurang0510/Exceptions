"""
News2Trade AI - Trading Signal ML Model
Trains a Random Forest classifier on engineered features to predict Buy / Sell / Hold.
Provides explainability via feature importances and per-prediction reasoning.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    BEGINNER_CONFIDENCE_THRESHOLD,
    HYPE_SCORE_THRESHOLD,
)


class TradingSignalModel:
    """
    ML model that combines sentiment features, hype features, and article
    metadata to predict Buy / Sell / Hold with explainability.
    """

    FEATURE_NAMES = [
        "sentiment_score",
        "sentiment_confidence",
        "positive_prob",
        "negative_prob",
        "neutral_prob",
        "hype_score",
        "punctuation_score",
        "subjectivity",
        "source_trusted",
        "text_length",
        "title_length",
        "has_numbers",
    ]

    LABELS = ["Buy", "Hold", "Sell"]

    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.LABELS)
        self.is_trained = False

    # ─── Feature Engineering ──────────────────────────────────────

    @staticmethod
    def extract_features(
        text: str,
        title: str,
        sentiment_result: dict,
        hype_result: dict,
    ) -> np.ndarray:
        """
        Build the feature vector from sentiment + hype analysis results.
        """
        details = sentiment_result.get("details", {})

        features = [
            sentiment_result.get("sentiment_score", 0.0),
            sentiment_result.get("confidence", 0.0),
            details.get("positive", 0.0),
            details.get("negative", 0.0),
            details.get("neutral", 0.0),
            hype_result.get("hype_score", 0.0),
            hype_result.get("punctuation_score", 0.0),
            hype_result.get("subjectivity", 0.5),
            1.0 if hype_result.get("source_trusted", False) else 0.0,
            min(len(text) / 1000.0, 5.0),             # normalized text length
            min(len(title) / 200.0, 1.0),              # normalized title length
            1.0 if any(c.isdigit() for c in text) else 0.0,  # has numbers
        ]
        return np.array(features, dtype=np.float64)

    # ─── Training ─────────────────────────────────────────────────

    def train_on_dataset(self, df: pd.DataFrame, sentiment_results: list, hype_results: list):
        """
        Train the model using the analysed news dataset.
        Generates synthetic labels based on sentiment + hype signals (since we
        don't have ground-truth trading labels from the news API).
        """
        X, y = [], []

        for i, row in df.iterrows():
            if i >= len(sentiment_results) or i >= len(hype_results):
                break

            sent = sentiment_results[i]
            hype = hype_results[i]
            text = str(row.get("text", ""))
            title = str(row.get("title", ""))

            features = self.extract_features(text, title, sent, hype)
            X.append(features)

            # Generate training label from sentiment + hype
            label = self._generate_label(sent, hype)
            y.append(label)

        if len(X) < 5:
            print("[WARN] Not enough data to train. Using rule-based fallback.")
            self.is_trained = False
            return

        X = np.array(X)
        y_encoded = self.label_encoder.transform(y)

        # Train a Random Forest with good defaults
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced",
        )
        self.model.fit(X, y_encoded)
        self.is_trained = True

        # Cross-validation score
        if len(X) >= 10:
            cv_scores = cross_val_score(self.model, X, y_encoded, cv=min(5, len(X)), scoring="accuracy")
            print(f"[INFO] Model trained. CV Accuracy: {cv_scores.mean():.2%} (±{cv_scores.std():.2%})")
        else:
            print(f"[INFO] Model trained on {len(X)} samples.")

        # Save model
        os.makedirs(MODEL_PATH, exist_ok=True)
        joblib.dump(self.model, os.path.join(MODEL_PATH, "signal_model.pkl"))
        print(f"[INFO] Model saved to {MODEL_PATH}signal_model.pkl")

    def _generate_label(self, sentiment: dict, hype: dict) -> str:
        """
        Generate a synthetic Buy/Sell/Hold label from sentiment and hype signals.
        This simulates what ground-truth labels would look like:
          - Very positive sentiment + low hype → Buy
          - Very negative sentiment + low hype → Sell  
          - Neutral, mixed, or high hype → Hold
        """
        score = sentiment.get("sentiment_score", 0.0)
        conf = sentiment.get("confidence", 0.0)
        hype_score = hype.get("hype_score", 0.0)

        # High hype → always Hold (unreliable news)
        if hype_score >= HYPE_SCORE_THRESHOLD:
            return "Hold"

        # Strong positive signal → Buy
        if score >= 0.3 and conf >= 0.4:
            return "Buy"

        # Strong negative signal → Sell
        if score <= -0.3 and conf >= 0.4:
            return "Sell"

        # Everything else → Hold
        return "Hold"

    # ─── Prediction ───────────────────────────────────────────────

    def predict(
        self,
        text: str,
        title: str,
        sentiment_result: dict,
        hype_result: dict,
        beginner_mode: bool = False,
    ) -> dict:
        """
        Predict Buy / Sell / Hold and provide explanation.

        Returns:
            {
                "signal":       str (Buy / Sell / Hold),
                "confidence":   float [0, 1],
                "probabilities": dict {Buy: ..., Sell: ..., Hold: ...},
                "explanation":  str,
                "features_used": dict,
                "beginner_override": bool,
                "risk_level":   str (Low / Medium / High),
            }
        """
        features = self.extract_features(text, title, sentiment_result, hype_result)

        if self.is_trained and self.model is not None:
            return self._predict_ml(features, sentiment_result, hype_result, beginner_mode)
        else:
            return self._predict_rule_based(features, sentiment_result, hype_result, beginner_mode)

    def _predict_ml(self, features, sentiment, hype, beginner_mode) -> dict:
        """Use the trained model for prediction."""
        X = features.reshape(1, -1)
        proba = self.model.predict_proba(X)[0]

        # Map probabilities to labels
        prob_dict = {}
        for i, label in enumerate(self.label_encoder.classes_):
            label_name = self.LABELS[i] if i < len(self.LABELS) else f"Class_{i}"
            prob_dict[label_name] = round(float(proba[i]), 4)

        # Get predicted class
        pred_idx = np.argmax(proba)
        signal = self.LABELS[pred_idx]
        confidence = float(proba[pred_idx])

        # Apply beginner mode
        beginner_override = False
        threshold = BEGINNER_CONFIDENCE_THRESHOLD if beginner_mode else CONFIDENCE_THRESHOLD

        if confidence < threshold:
            original_signal = signal
            signal = "Hold"
            beginner_override = True

        # Hype override
        if hype.get("is_hype", False):
            signal = "Hold"
            beginner_override = True

        # Risk level
        risk_level = self._assess_risk(confidence, hype, beginner_mode)

        # Explanation
        explanation = self._build_explanation(
            signal, confidence, sentiment, hype, beginner_mode, beginner_override,
            prob_dict, method="ML Model"
        )

        # Feature importance dict
        feat_importance = {}
        if hasattr(self.model, "feature_importances_"):
            for name, imp in zip(self.FEATURE_NAMES, self.model.feature_importances_):
                feat_importance[name] = round(float(imp), 4)

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "explanation": explanation,
            "features_used": feat_importance,
            "beginner_override": beginner_override,
            "risk_level": risk_level,
        }

    def _predict_rule_based(self, features, sentiment, hype, beginner_mode) -> dict:
        """Fallback rule-based prediction when model isn't trained."""
        score = sentiment.get("sentiment_score", 0.0)
        conf = sentiment.get("confidence", 0.0)
        hype_score = hype.get("hype_score", 0.0)
        is_hype = hype.get("is_hype", False)

        # Determine signal from rules
        if is_hype:
            signal = "Hold"
            confidence = 0.5
        elif score >= 0.3 and conf >= 0.4:
            signal = "Buy"
            confidence = min(0.5 + score * 0.4 + conf * 0.2, 0.95)
        elif score <= -0.3 and conf >= 0.4:
            signal = "Sell"
            confidence = min(0.5 + abs(score) * 0.4 + conf * 0.2, 0.95)
        else:
            signal = "Hold"
            confidence = max(0.4, 1.0 - abs(score) * 0.5)

        prob_dict = {"Buy": 0.0, "Sell": 0.0, "Hold": 0.0}
        prob_dict[signal] = round(confidence, 4)
        remaining = 1.0 - confidence
        for k in prob_dict:
            if k != signal:
                prob_dict[k] = round(remaining / 2, 4)

        # Beginner mode
        beginner_override = False
        threshold = BEGINNER_CONFIDENCE_THRESHOLD if beginner_mode else CONFIDENCE_THRESHOLD
        if confidence < threshold and signal != "Hold":
            signal = "Hold"
            beginner_override = True

        risk_level = self._assess_risk(confidence, hype, beginner_mode)

        explanation = self._build_explanation(
            signal, confidence, sentiment, hype, beginner_mode, beginner_override,
            prob_dict, method="Rule-Based"
        )

        return {
            "signal": signal,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "explanation": explanation,
            "features_used": {"method": "rule-based (no trained model)"},
            "beginner_override": beginner_override,
            "risk_level": risk_level,
        }

    # ─── Explainability ──────────────────────────────────────────

    def _build_explanation(self, signal, confidence, sentiment, hype,
                           beginner_mode, beginner_override, probs, method):
        """Build a human-readable explanation of the prediction."""
        lines = []
        lines.append(f"📊 Signal: **{signal}** (Confidence: {confidence:.0%})")
        lines.append(f"🔍 Method: {method}")
        lines.append("")

        # Sentiment reasoning
        s_score = sentiment.get("sentiment_score", 0.0)
        s_label = sentiment.get("sentiment_label", "neutral")
        lines.append(f"📰 Sentiment: {s_label.upper()} (score: {s_score:+.2f})")

        # Hype reasoning
        h_score = hype.get("hype_score", 0.0)
        if hype.get("is_hype"):
            lines.append(f"⚠️ HYPE DETECTED (score: {h_score:.0%}) — Signal forced to HOLD")
            for flag in hype.get("flags", []):
                lines.append(f"   • {flag}")
        else:
            lines.append(f"✅ Hype check passed (score: {h_score:.0%})")

        # Beginner mode
        if beginner_mode:
            lines.append("")
            lines.append("🛡️ Beginner Safety Mode: ON")
            if beginner_override:
                lines.append("   → Signal changed to HOLD (confidence below safety threshold)")
            else:
                lines.append("   → Signal meets safety threshold")

        # Probabilities
        lines.append("")
        lines.append("📈 Probability Breakdown:")
        for label, prob in sorted(probs.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            lines.append(f"   {label:5s} {bar} {prob:.1%}")

        return "\n".join(lines)

    def _assess_risk(self, confidence, hype, beginner_mode) -> str:
        """Assess overall risk level."""
        hype_score = hype.get("hype_score", 0.0)

        if hype.get("is_hype") or confidence < 0.4:
            return "High"
        elif confidence < 0.6 or hype_score > 0.4:
            return "Medium"
        else:
            return "Low"

    # ─── Model Persistence ───────────────────────────────────────

    def load_model(self, path: str = None):
        """Load a previously saved model."""
        if path is None:
            path = os.path.join(MODEL_PATH, "signal_model.pkl")
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.is_trained = True
            print(f"[INFO] Model loaded from {path}")
        else:
            print(f"[INFO] No saved model found at {path}. Will train fresh.")

    def get_feature_importances(self) -> dict:
        """Return feature importances if model is trained."""
        if self.is_trained and hasattr(self.model, "feature_importances_"):
            return {
                name: round(float(imp), 4)
                for name, imp in zip(self.FEATURE_NAMES, self.model.feature_importances_)
            }
        return {}
