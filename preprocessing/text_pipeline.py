"""
NLP Preprocessing Pipeline
===========================
A modular, research-grade text preprocessing pipeline tailored for
financial news.  Each stage is an independent function so stages can
be composed, swapped, or ablated during experimentation.

Pipeline stages
---------------
1. **Text normalisation** — lowercase, unicode, whitespace collapse
2. **Punctuation removal** — strip non-alphanumeric symbols
3. **Tokenisation** — word-level splitting via NLTK
4. **Stopword removal** — English + financial stop-words
5. **Lemmatisation** — WordNet lemmatiser
6. **Financial keyword extraction** — domain-specific lexicon match
7. **Named-entity recognition** — spaCy NER for ORG / MONEY / PERCENT
"""

import re
import logging
from typing import List, Optional

import pandas as pd
import nltk
from pathlib import Path as _Path

# Point NLTK to local data directory first
_nltk_local = _Path(__file__).resolve().parent.parent / "nltk_data"
if _nltk_local.exists():
    nltk.data.path.insert(0, str(_nltk_local))

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ── One-time NLTK downloads (silent) ──────────────────────────────
for _pkg in ("punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger_eng"):
    nltk.download(_pkg, quiet=True)

_LEMMATIZER = WordNetLemmatizer()
_STOP_WORDS = set(stopwords.words("english"))

# Extra finance stop-words that carry little discriminative signal
_FINANCE_STOP = {
    "stock", "market", "share", "shares", "company", "inc", "corp",
    "ltd", "said", "today", "also", "new", "year", "quarter",
}

FINANCIAL_KEYWORDS = {
    "earnings", "revenue", "profit", "loss", "dividend", "merger",
    "acquisition", "ipo", "bankruptcy", "regulation", "inflation",
    "recession", "bull", "bear", "rally", "crash", "volatility",
    "growth", "decline", "upgrade", "downgrade", "buyback",
    "guidance", "forecast", "target", "overweight", "underweight",
    "sec", "fed", "interest rate", "gdp", "cpi", "unemployment",
    "hedge", "short", "squeeze", "options", "derivatives",
}


# ────────────────────────────────────────────────────────────────────
# 1. Text normalisation
# ────────────────────────────────────────────────────────────────────
def normalise_text(text: str) -> str:
    """Lowercase, strip unicode artefacts, collapse whitespace."""
    text = text.lower().strip()
    text = text.encode("ascii", "ignore").decode()  # drop non-ASCII
    text = re.sub(r"\s+", " ", text)
    return text


# ────────────────────────────────────────────────────────────────────
# 2. Punctuation removal
# ────────────────────────────────────────────────────────────────────
def remove_punctuation(text: str) -> str:
    """Remove all non-alphanumeric characters except spaces."""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


# ────────────────────────────────────────────────────────────────────
# 3. Tokenisation
# ────────────────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    """NLTK word-level tokenisation."""
    return word_tokenize(text)


# ────────────────────────────────────────────────────────────────────
# 4. Stopword removal
# ────────────────────────────────────────────────────────────────────
def remove_stopwords(tokens: List[str], include_finance: bool = True) -> List[str]:
    """Remove English and (optionally) financial stop-words."""
    sw = _STOP_WORDS | _FINANCE_STOP if include_finance else _STOP_WORDS
    return [t for t in tokens if t not in sw and len(t) > 1]


# ────────────────────────────────────────────────────────────────────
# 5. Lemmatisation
# ────────────────────────────────────────────────────────────────────
def lemmatize(tokens: List[str]) -> List[str]:
    """WordNet lemmatisation."""
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


# ────────────────────────────────────────────────────────────────────
# 6. Financial keyword extraction
# ────────────────────────────────────────────────────────────────────
def extract_financial_keywords(text: str) -> List[str]:
    """Return financial keywords found in *text*."""
    text_lower = text.lower()
    return [kw for kw in FINANCIAL_KEYWORDS if kw in text_lower]


# ────────────────────────────────────────────────────────────────────
# 7. Named-entity recognition (lazy-loaded spaCy)
# ────────────────────────────────────────────────────────────────────
_NLP = None


def _get_spacy():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_sm")
        except (OSError, ImportError):
            logger.warning(
                "spaCy not available. "
                "Run:  pip install spacy && python -m spacy download en_core_web_sm"
            )
            _NLP = False
    return _NLP


def extract_entities(text: str) -> List[dict]:
    """Extract named entities using spaCy (ORG, MONEY, PERCENT, GPE)."""
    nlp = _get_spacy()
    if not nlp:
        return []
    doc = nlp(text)
    return [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ in ("ORG", "MONEY", "PERCENT", "GPE", "PERSON")
    ]


# ────────────────────────────────────────────────────────────────────
# Composite pipeline
# ────────────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Run the full preprocessing pipeline and return a clean string."""
    text = normalise_text(text)
    text = remove_punctuation(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "title",
    output_col: str = "clean_text",
) -> pd.DataFrame:
    """Apply full pipeline to a DataFrame column."""
    df = df.copy()
    df[output_col] = df[text_col].astype(str).apply(preprocess_text)
    df["financial_keywords"] = df[text_col].astype(str).apply(
        extract_financial_keywords
    )
    df["n_financial_keywords"] = df["financial_keywords"].apply(len)
    logger.info("Preprocessed %d rows.", len(df))
    return df
