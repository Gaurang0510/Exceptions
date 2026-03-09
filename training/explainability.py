"""
Explainable AI Module
======================
Provides model interpretability via:

1. **SHAP value analysis** — game-theoretic feature attributions
2. **Feature importance** — built-in model importances + permutation
3. **Attention visualisation** — transformer attention heatmaps

Interpretability is critical in financial AI: regulators, portfolio
managers, and risk officers need to understand *why* a model predicts
a particular stock movement before acting on it.
"""

import logging
from typing import Optional, Any, List

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from data.dataset_builder import LABEL_MAP

logger = logging.getLogger(__name__)
CLASS_NAMES = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]


# ════════════════════════════════════════════════════════════════════
# 1. SHAP VALUE ANALYSIS
# ════════════════════════════════════════════════════════════════════
def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_samples: int = 500,
) -> shap.Explanation:
    """Compute SHAP values using the appropriate explainer.

    Automatically selects TreeExplainer (tree models) or
    KernelExplainer (black-box) based on model type.
    """
    sample = X[:max_samples] if len(X) > max_samples else X

    try:
        # Tree-based models (RF, XGBoost)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
    except Exception:
        # Fallback to KernelExplainer
        logger.info("Using KernelExplainer (slower) for SHAP.")
        background = shap.kmeans(X, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(sample)

    return shap_values


def plot_shap_summary(
    shap_values: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    class_idx: int = 2,  # UP class
    max_display: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """SHAP beeswarm / summary plot for a single class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    vals = shap_values[class_idx] if isinstance(shap_values, list) else shap_values
    shap.summary_plot(
        vals,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(f"SHAP Summary — {CLASS_NAMES[class_idx]} class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gcf()


def plot_shap_bar(
    shap_values: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display: int = 15,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Global SHAP feature importance bar chart."""
    if isinstance(shap_values, list):
        # Average absolute SHAP across classes
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs = np.abs(shap_values).mean(axis=0)

    n_feats = min(max_display, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-n_feats:][::-1]
    names = (
        [feature_names[i] for i in top_idx]
        if feature_names
        else [f"Feature {i}" for i in top_idx]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(n_feats), mean_abs[top_idx][::-1], color="#2196F3")
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("Global Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ════════════════════════════════════════════════════════════════════
# 2. BUILT-IN FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════
def plot_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart from tree-model ``feature_importances_``."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        logger.warning("Model has no feature_importances_ attribute.")
        return plt.figure()

    n_feats = min(top_n, len(importances))
    top_idx = np.argsort(importances)[-n_feats:][::-1]
    names = (
        [feature_names[i] for i in top_idx]
        if feature_names
        else [f"Feature {i}" for i in top_idx]
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(n_feats), importances[top_idx][::-1], color="#FF9800")
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title("Feature Importance (MDI)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


# ════════════════════════════════════════════════════════════════════
# 3. ATTENTION MAP VISUALISATION (Transformer)
# ════════════════════════════════════════════════════════════════════
def plot_attention_map(
    text: str,
    model=None,
    tokenizer=None,
    layer: int = -1,
    head: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Visualise self-attention weights for a single headline.

    Works with HuggingFace transformer models that return
    ``output_attentions=True``.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        from config.settings import SENTIMENT_MODEL

        tokenizer = tokenizer or AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
        model = model or AutoModel.from_pretrained(SENTIMENT_MODEL, output_attentions=True)
        model.eval()

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        attn = outputs.attentions[layer][0, head].numpy()  # (seq, seq)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attn, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=8)
        ax.set_yticklabels(tokens, fontsize=8)
        ax.set_title(f"Self-Attention (Layer {layer}, Head {head})",
                      fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        return fig

    except Exception as exc:
        logger.warning("Attention map generation failed: %s", exc)
        return plt.figure()


# ════════════════════════════════════════════════════════════════════
# Composite explainability report
# ════════════════════════════════════════════════════════════════════
def generate_explainability_report(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> dict:
    """Generate a full explainability report: SHAP + feature importance."""
    report = {}

    # SHAP
    try:
        shap_vals = compute_shap_values(model, X, feature_names)
        report["shap_values"] = shap_vals
        report["shap_summary_fig"] = plot_shap_summary(shap_vals, X, feature_names)
        report["shap_bar_fig"] = plot_shap_bar(shap_vals, X, feature_names)
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)

    # Built-in importance
    report["importance_fig"] = plot_feature_importance(model, feature_names)

    return report
