"""
Training Pipeline
==================
Professional ML training workflow with:

- train / test split
- k-fold cross-validation
- hyper-parameter tuning (GridSearchCV)
- feature scaling
- model checkpointing (joblib)
- comparative benchmarking of all models
"""

import logging
import time
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from config.settings import (
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    MODEL_DIR,
)
from models.classical_models import MODEL_REGISTRY, PARAM_GRIDS, get_model

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════
# Feature scaling
# ════════════════════════════════════════════════════════════════════
def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train, transform both splits."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


# ════════════════════════════════════════════════════════════════════
# Single-model training
# ════════════════════════════════════════════════════════════════════
def train_single_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: bool = True,
) -> Any:
    """Train one classical model, optionally with hyper-parameter search."""
    logger.info("Training: %s (tune=%s)", model_name, tune)
    model = get_model(model_name)

    if tune and model_name in PARAM_GRIDS:
        grid = GridSearchCV(
            model,
            PARAM_GRIDS[model_name],
            cv=CV_FOLDS,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        logger.info("Best params for %s: %s (score=%.4f)",
                     model_name, grid.best_params_, grid.best_score_)
        return grid.best_estimator_

    model.fit(X_train, y_train)
    return model


# ════════════════════════════════════════════════════════════════════
# Cross-validation
# ════════════════════════════════════════════════════════════════════
def cross_validate_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = CV_FOLDS,
) -> Dict[str, float]:
    """Stratified k-fold cross-validation summary."""
    model = get_model(model_name)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=-1)
    result = {
        "model": model_name,
        "cv_f1_mean": scores.mean(),
        "cv_f1_std": scores.std(),
    }
    logger.info("CV %s — F1: %.4f (±%.4f)", model_name, scores.mean(), scores.std())
    return result


# ════════════════════════════════════════════════════════════════════
# Save / load model checkpoints
# ════════════════════════════════════════════════════════════════════
def save_model(model: Any, name: str, scaler: Optional[StandardScaler] = None) -> Path:
    """Persist model (and optionally scaler) to disk."""
    path = MODEL_DIR / f"{name}.joblib"
    payload = {"model": model, "scaler": scaler}
    joblib.dump(payload, path)
    logger.info("Model saved → %s", path)
    return path


def load_model(name: str) -> Dict[str, Any]:
    """Load a saved model checkpoint."""
    path = MODEL_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint found at {path}")
    payload = joblib.load(path)
    logger.info("Model loaded ← %s", path)
    return payload


# ════════════════════════════════════════════════════════════════════
# Full benchmark pipeline
# ════════════════════════════════════════════════════════════════════
def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    model_names: Optional[list] = None,
    tune: bool = True,
    test_size: int | float = TEST_SIZE,
) -> pd.DataFrame:
    """Train, evaluate, and compare all classical models.

    Args:
        X: Feature matrix
        y: Labels
        model_names: List of model names to train (default: all)
        tune: Whether to perform hyperparameter tuning
        test_size: If int, absolute number of test samples. If float, fraction.

    Returns a summary DataFrame sorted by macro-F1 score.
    """
    model_names = model_names or list(MODEL_REGISTRY.keys())

    # Handle test_size as absolute count or fraction
    if isinstance(test_size, int):
        actual_test_size = test_size / len(y) if test_size < len(y) else 0.2
        logger.info(f"Using {test_size} samples for testing ({actual_test_size:.1%} of data)")
    else:
        actual_test_size = test_size
        logger.info(f"Using {actual_test_size:.1%} of data for testing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=actual_test_size, stratify=y, random_state=RANDOM_STATE,
    )
    
    logger.info(f"Train set: {len(y_train)} samples | Test set: {len(y_test)} samples")
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    results = []
    for name in model_names:
        t0 = time.time()
        try:
            model = train_single_model(name, X_train_s, y_train, tune=tune)
            y_pred = model.predict(X_test_s)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            report = classification_report(y_test, y_pred, output_dict=True)

            # Save checkpoint
            save_model(model, name, scaler=scaler)

            elapsed = time.time() - t0
            results.append({
                "model": name,
                "accuracy": acc,
                "f1_macro": f1,
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "train_time_s": round(elapsed, 2),
            })
            logger.info("%s — Acc: %.4f | F1: %.4f | Time: %.1fs",
                        name, acc, f1, elapsed)
        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc)
            results.append({"model": name, "error": str(exc)})

    summary = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    logger.info("Benchmark summary:\n%s", summary.to_string(index=False))
    return summary
