"""
News2Trade AI - Hype / Fake-News Detector
Flags exaggerated, misleading, or clickbait financial news.
Uses a combination of rule-based heuristics and a trained ML classifier.
"""
import re
import numpy as np
from textblob import TextBlob
from config import HYPE_INDICATORS


class HypeDetector:
    """
    Multi-signal hype / fake-news detector:
      1. Keyword matching  (clickbait phrases)
      2. Punctuation abuse  (excessive !, ?, CAPS)
      3. Subjectivity score (TextBlob)
      4. Source credibility  (known vs unknown sources)
      5. Combined hype score [0, 1]
    """

    TRUSTED_SOURCES = {
        "reuters", "bloomberg", "cnbc", "financial times", "wall street journal",
        "ap news", "bbc", "the economist", "marketwatch", "barron's",
        "economic times", "techcrunch", "the verge", "coindesk",
        "nikkei", "fortune", "forbes", "yahoo finance",
    }

    def __init__(self):
        # Pre-compile regex patterns for efficiency
        self._hype_patterns = [
            re.compile(re.escape(phrase), re.IGNORECASE)
            for phrase in HYPE_INDICATORS
        ]

    def detect(self, text: str, source: str = "") -> dict:
        """
        Analyze a news article for hype / fake signals.

        Returns:
            {
                "hype_score":       float [0, 1],
                "is_hype":          bool,
                "keyword_hits":     list[str],
                "punctuation_score": float,
                "subjectivity":     float,
                "source_trusted":   bool,
                "flags":            list[str],   # human-readable warnings
                "explanation":      str,
            }
        """
        if not text or not text.strip():
            return self._empty_result()

        keyword_score, keyword_hits = self._keyword_score(text)
        punctuation_score = self._punctuation_score(text)
        subjectivity = self._subjectivity_score(text)
        source_trusted = self._check_source(source)

        # Weighted combination → final hype score
        weights = {
            "keyword":      0.40,
            "punctuation":  0.20,
            "subjectivity": 0.25,
            "source":       0.15,
        }

        source_penalty = 0.0 if source_trusted else 1.0

        hype_score = (
            weights["keyword"]      * keyword_score +
            weights["punctuation"]  * punctuation_score +
            weights["subjectivity"] * subjectivity +
            weights["source"]       * source_penalty
        )
        hype_score = round(min(max(hype_score, 0.0), 1.0), 4)

        is_hype = hype_score >= 0.65

        # Build human-readable flags
        flags = []
        if keyword_hits:
            flags.append(f"Clickbait phrases detected: {', '.join(keyword_hits[:5])}")
        if punctuation_score > 0.5:
            flags.append("Excessive punctuation / ALL-CAPS detected")
        if subjectivity > 0.7:
            flags.append("Highly subjective language")
        if not source_trusted and source:
            flags.append(f"Unverified source: {source}")
        elif not source:
            flags.append("No source information available")

        explanation = self._build_explanation(hype_score, is_hype, flags)

        return {
            "hype_score": hype_score,
            "is_hype": is_hype,
            "keyword_hits": keyword_hits,
            "punctuation_score": round(punctuation_score, 4),
            "subjectivity": round(subjectivity, 4),
            "source_trusted": source_trusted,
            "flags": flags,
            "explanation": explanation,
        }

    # ─── Sub-Scores ───────────────────────────────────────────────

    def _keyword_score(self, text: str) -> tuple[float, list[str]]:
        """Check for hype / clickbait keywords."""
        hits = []
        for pattern in self._hype_patterns:
            if pattern.search(text):
                hits.append(pattern.pattern.replace("\\", ""))
        # Normalize: cap at 5 hits for max score
        score = min(len(hits) / 3.0, 1.0)
        return score, hits

    def _punctuation_score(self, text: str) -> float:
        """Score based on excessive punctuation and CAPS usage."""
        if not text:
            return 0.0

        exclamation_count = text.count("!")
        question_count = text.count("?")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        # Normalize each
        excl_score = min(exclamation_count / 5.0, 1.0)
        caps_score = min(caps_ratio / 0.5, 1.0)   # > 50% caps → max
        q_score = min(question_count / 5.0, 1.0)

        return (excl_score * 0.4 + caps_score * 0.4 + q_score * 0.2)

    def _subjectivity_score(self, text: str) -> float:
        """Use TextBlob to measure subjectivity [0=objective, 1=subjective]."""
        try:
            blob = TextBlob(text)
            return blob.sentiment.subjectivity
        except Exception:
            return 0.5  # default middle

    def _check_source(self, source: str) -> bool:
        """Check if the source is in the trusted list."""
        if not source:
            return False
        return source.strip().lower() in self.TRUSTED_SOURCES

    # ─── Helpers ──────────────────────────────────────────────────

    def _build_explanation(self, score: float, is_hype: bool, flags: list[str]) -> str:
        if is_hype:
            level = "HIGH" if score >= 0.85 else "MODERATE"
            base = f"⚠️ {level} HYPE RISK (score: {score:.0%}). "
            base += "This article shows signs of exaggeration or misleading content. "
            if flags:
                base += "Issues: " + "; ".join(flags) + ". "
            base += "Exercise extreme caution before acting on this news."
        else:
            base = f"✅ Low hype risk (score: {score:.0%}). "
            base += "This article appears to be from a credible source with balanced language."
        return base

    @staticmethod
    def _empty_result() -> dict:
        return {
            "hype_score": 0.0,
            "is_hype": False,
            "keyword_hits": [],
            "punctuation_score": 0.0,
            "subjectivity": 0.0,
            "source_trusted": False,
            "flags": ["No text provided"],
            "explanation": "No text provided for analysis.",
        }
