"""
Classical ML Models
====================
Baseline and tree-based classifiers wrapped in a uniform interface for
the training pipeline.

Models
------
- Logistic Regression (baseline)
- Support Vector Machine (baseline)
- Random Forest (ensemble)
- XGBoost (gradient boosting)
"""

import logging
from typing import Dict, Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from config.settings import RANDOM_STATE

logger = logging.getLogger(__name__)


def get_logistic_regression(params: Dict[str, Any] | None = None) -> LogisticRegression:
    """Multinomial logistic regression with L2 regularisation.

    Advantages
    ----------
    - Fast training; interpretable coefficients.
    - Strong baseline for text-classification with TF-IDF features.
    """
    defaults = dict(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced",  # Handle class imbalance
        random_state=RANDOM_STATE,
    )
    if params:
        defaults.update(params)
    return LogisticRegression(**defaults)


def get_svm(params: Dict[str, Any] | None = None) -> SVC:
    """Support Vector Machine with RBF kernel.

    Advantages
    ----------
    - Effective in high-dimensional feature spaces.
    - Probability calibration via Platt scaling.
    """
    defaults = dict(
        kernel="rbf",
        C=10.0,  # Higher C for less regularization
        gamma="scale",
        probability=True,
        class_weight="balanced",  # Handle class imbalance
        random_state=RANDOM_STATE,
    )
    if params:
        defaults.update(params)
    return SVC(**defaults)


def get_random_forest(params: Dict[str, Any] | None = None) -> RandomForestClassifier:
    """Random Forest ensemble.

    Advantages
    ----------
    - Handles non-linear interactions; robust to outliers.
    - Built-in feature importance (mean decrease in impurity).
    """
    defaults = dict(
        n_estimators=500,
        max_depth=20,  # Limit depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",  # Good for high-dimensional data
        class_weight="balanced",  # Handle class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if params:
        defaults.update(params)
    return RandomForestClassifier(**defaults)


def get_xgboost(params: Dict[str, Any] | None = None):
    """XGBoost gradient-boosted tree classifier.

    Advantages
    ----------
    - State-of-the-art tabular performance.
    - Regularisation (L1/L2), native handling of missing values.
    - GPU acceleration when available.
    """
    from xgboost import XGBClassifier

    # Check for GPU availability using XGBoost's native CUDA detection
    gpu_available = False
    try:
        import xgboost as xgb
        # XGBoost 2.0+ can detect GPU support natively
        gpu_available = xgb.build_info().get('USE_CUDA', False)
    except Exception:
        pass

    defaults = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # Enable GPU with memory-safe settings if available
    if gpu_available:
        defaults.update({
            "device": "cuda",
            "tree_method": "hist",  # GPU-accelerated histogram
            "max_bin": 256,         # Limit memory usage
        })
        logger.info("XGBoost: GPU acceleration enabled (native CUDA)")
    else:
        defaults["tree_method"] = "hist"  # Fast CPU method
        logger.info("XGBoost: Using CPU (GPU not available)")

    if params:
        defaults.update(params)
    return XGBClassifier(**defaults)


# ────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "logistic_regression": get_logistic_regression,
    "svm": get_svm,
    "random_forest": get_random_forest,
    "xgboost": get_xgboost,
}


def get_model(name: str, params: Dict[str, Any] | None = None):
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](params)


# ────────────────────────────────────────────────────────────────────
# Hyper-parameter search spaces (for tuning)
# ────────────────────────────────────────────────────────────────────
PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0, 100.0],
    },
    "svm": {
        "C": [1.0, 10.0, 50.0, 100.0],
        "gamma": ["scale", 0.01, 0.001],
    },
    "random_forest": {
        "n_estimators": [300, 500, 700],
        "max_depth": [15, 20, 30],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    },
    "xgboost": {
        "n_estimators": [300, 500, 700],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 1.0],
    },
}
