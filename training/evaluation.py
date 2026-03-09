"""
Model Evaluation & Metrics
===========================
Rigorous evaluation suite for financial prediction models.

Metrics
-------
- Accuracy, Precision, Recall, F1 Score (per-class & macro)
- ROC-AUC (one-vs-rest)
- Confusion Matrix

All functions return both raw numbers and publication-quality
matplotlib figures.
"""

import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

from data.dataset_builder import LABEL_MAP

logger = logging.getLogger(__name__)

CLASS_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]  # DOWN, NEUTRAL, UP


# ════════════════════════════════════════════════════════════════════
# Core metrics
# ════════════════════════════════════════════════════════════════════
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute classification metrics and return a summary dict."""
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Per-class metrics
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    metrics["per_class"] = report

    # ROC-AUC (requires probability estimates)
    if y_prob is not None:
        try:
            y_bin = label_binarize(y_true, classes=[0, 1, 2])
            metrics["roc_auc_macro"] = roc_auc_score(
                y_bin, y_prob, average="macro", multi_class="ovr",
            )
        except ValueError:
            metrics["roc_auc_macro"] = None
    else:
        metrics["roc_auc_macro"] = None

    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    logger.info(
        "Metrics — Acc: %.4f | F1: %.4f | AUC: %s",
        metrics["accuracy"],
        metrics["f1_macro"],
        f'{metrics["roc_auc_macro"]:.4f}' if metrics["roc_auc_macro"] else "N/A",
    )
    return metrics


# ════════════════════════════════════════════════════════════════════
# Confusion matrix heatmap
# ════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Publication-quality confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ════════════════════════════════════════════════════════════════════
# ROC curves (one-vs-rest)
# ════════════════════════════════════════════════════════════════════
def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curves (One-vs-Rest)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-class ROC curves with AUC annotations."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(8, 6))

    colours = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, (name, colour) in enumerate(zip(CLASS_NAMES, colours)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colour, lw=2,
                label=f"{name} (AUC = {roc_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ════════════════════════════════════════════════════════════════════
# Model comparison bar chart
# ════════════════════════════════════════════════════════════════════
def plot_model_comparison(
    summary_df: pd.DataFrame,
    metric: str = "f1_macro",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart comparing models by a chosen metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = sns.color_palette("viridis", len(summary_df))
    bars = ax.barh(summary_df["model"], summary_df[metric], color=colours)
    ax.set_xlabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=5)
    ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ════════════════════════════════════════════════════════════════════
# Quick evaluation helper
# ════════════════════════════════════════════════════════════════════
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
) -> Dict[str, Any]:
    """One-call evaluation: metrics + confusion matrix + ROC."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["model_name"] = model_name

    # Generate plots
    cm_fig = plot_confusion_matrix(y_test, y_pred, title=f"{model_name} — Confusion Matrix")
    metrics["confusion_matrix_fig"] = cm_fig

    if y_prob is not None:
        roc_fig = plot_roc_curves(y_test, y_prob, title=f"{model_name} — ROC Curves")
        metrics["roc_fig"] = roc_fig

    return metrics
