"""
explainability.py
=================
Explainable AI (XAI) module for Tanzania BEST education performance project.
Implements permutation importance, partial dependence plots, ICE plots,
and SHAP-style manual approximation (when SHAP is unavailable).

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.inspection import (
    PartialDependenceDisplay,
    permutation_importance,
)

logger = logging.getLogger("explainability")

PLOT_STYLE = {
    "figure.dpi": 150,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
}
plt.rcParams.update(PLOT_STYLE)


# ---------------------------------------------------------------------------
# Feature importance plots
# ---------------------------------------------------------------------------
def plot_feature_importance(
    model: BaseEstimator,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot intrinsic feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"{model_name} has no feature_importances_ attribute.")
        return None

    fi = pd.Series(model.feature_importances_, index=feature_names)
    fi = fi.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, max(5, len(fi) * 0.35)))
    colours = ["steelblue" if v >= fi.quantile(0.5) else "lightsteelblue"
               for v in fi.values]
    ax.barh(fi.index, fi.values, color=colours, alpha=0.9)
    ax.axvline(fi.mean(), color="crimson", lw=1.5, linestyle="--",
               label=f"Mean = {fi.mean():.4f}")
    ax.set_xlabel("Feature Importance (impurity reduction)")
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_permutation_importance(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 20,
    n_repeats: int = 30,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """Permutation importance — model-agnostic and robust to correlated features."""
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats, random_state=42,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame({
        "feature":         feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    }).sort_values("importance_mean", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, max(5, len(perm_df) * 0.38)))
    ax.barh(perm_df["feature"], perm_df["importance_mean"],
            xerr=perm_df["importance_std"], color="darkorange", alpha=0.85,
            error_kw={"elinewidth": 1, "ecolor": "gray", "capsize": 3})
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Increase in MAE when feature is shuffled")
    ax.set_title(f"Permutation Importance — {model_name} (Test Set)", fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, perm_df


def plot_coefficients(
    model: BaseEstimator,
    feature_names: List[str],
    model_name: str = "Model",
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of standardised linear model coefficients."""
    if not hasattr(model, "coef_"):
        logger.warning(f"{model_name} has no coef_ attribute.")
        return None

    coef = pd.Series(model.coef_, index=feature_names)
    coef = coef.reindex(coef.abs().sort_values(ascending=True).tail(top_n).index)

    fig, ax = plt.subplots(figsize=(9, max(5, len(coef) * 0.38)))
    colours = ["steelblue" if v > 0 else "salmon" for v in coef.values]
    ax.barh(coef.index, coef.values, color=colours, alpha=0.85)
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Standardised Coefficient")
    ax.set_title(f"{model_name} Coefficients\n"
                 "(blue = positive association with pass rate)", fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_partial_dependence(
    model: BaseEstimator,
    X_train: np.ndarray,
    feature_names: List[str],
    features_to_plot: Optional[List[int]] = None,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Partial Dependence Plots for selected features."""
    if features_to_plot is None:
        features_to_plot = list(range(min(6, len(feature_names))))

    n_cols = 3
    n_rows = (len(features_to_plot) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()

    for idx, feat_idx in enumerate(features_to_plot):
        try:
            display = PartialDependenceDisplay.from_estimator(
                model, X_train, [feat_idx],
                feature_names=feature_names,
                ax=axes[idx], kind="average",
            )
            axes[idx].set_title(f"PDP: {feature_names[feat_idx]}", fontsize=9)
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"PDP unavailable\n{str(e)[:60]}",
                           ha="center", va="center", fontsize=8)
            axes[idx].set_visible(True)

    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Partial Dependence Plots — {model_name}", fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ice_curves(
    model: BaseEstimator,
    X_train: np.ndarray,
    feature_names: List[str],
    feature_idx: int = 0,
    n_ice_lines: int = 30,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Individual Conditional Expectation (ICE) plot for a single feature."""
    feat_name = feature_names[feature_idx]
    feat_vals = np.linspace(
        X_train[:, feature_idx].min(),
        X_train[:, feature_idx].max(),
        50,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sample_idx = np.random.choice(len(X_train), min(n_ice_lines, len(X_train)), replace=False)
    pdp_avg = np.zeros(len(feat_vals))

    for i, obs_idx in enumerate(sample_idx):
        X_ice = X_train[[obs_idx], :].repeat(len(feat_vals), axis=0).copy()
        X_ice[:, feature_idx] = feat_vals
        try:
            preds = model.predict(X_ice)
            ax.plot(feat_vals, preds, alpha=0.15, color="steelblue", lw=0.8)
            pdp_avg += preds
        except Exception:
            pass

    pdp_avg /= max(len(sample_idx), 1)
    ax.plot(feat_vals, pdp_avg, color="crimson", lw=2.5, label="PDP average")
    ax.set_xlabel(feat_name)
    ax.set_ylabel("Predicted CSEE Pass Rate (%)")
    ax.set_title(f"ICE Plot — {feat_name}\n({model_name})", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def xai_summary_report(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    model_name: str = "Model",
    output_dir: str = "outputs/figures",
) -> Dict[str, pd.DataFrame]:
    """
    Run the full XAI pipeline: feature importance, permutation importance,
    PDP, ICE plots. Saves all figures and returns summary DataFrames.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results: Dict[str, pd.DataFrame] = {}

    # Feature importance (tree models)
    fig = plot_feature_importance(model, feature_names, model_name,
                                  save_path=f"{output_dir}/xai_feat_importance.png")

    # Permutation importance
    fig_perm, perm_df = plot_permutation_importance(
        model, X_test, y_test, feature_names, model_name,
        save_path=f"{output_dir}/xai_permutation_importance.png",
    )
    results["permutation_importance"] = perm_df

    # Coefficients (linear models)
    fig_coef = plot_coefficients(model, feature_names, model_name,
                                 save_path=f"{output_dir}/xai_coefficients.png")

    # Top features for PDP
    if hasattr(model, "feature_importances_"):
        fi = pd.Series(model.feature_importances_, index=range(len(feature_names)))
        top6 = fi.nlargest(6).index.tolist()
    else:
        top6 = list(range(min(6, len(feature_names))))

    fig_pdp = plot_partial_dependence(
        model, X_test, feature_names, top6, model_name,
        save_path=f"{output_dir}/xai_pdp.png",
    )

    # ICE for the most important feature
    if top6:
        fig_ice = plot_ice_curves(
            model, X_test, feature_names, top6[0], model_name=model_name,
            save_path=f"{output_dir}/xai_ice.png",
        )

    logger.info("XAI report complete.")
    return results
