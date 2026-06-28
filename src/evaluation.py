"""
evaluation.py
=============
Model evaluation, diagnostic plots, learning curves, and residual analysis
for Tanzania BEST education performance project.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import learning_curve, validation_curve

logger = logging.getLogger("evaluation")

PLOT_STYLE = {
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
}
plt.rcParams.update(PLOT_STYLE)
sns.set_palette("tab10")


# ---------------------------------------------------------------------------
def actual_vs_predicted_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Publication-quality actual vs. predicted scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Left: scatter ----
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=50, alpha=0.72, color="steelblue",
               edgecolors="white", linewidth=0.4)
    lo = min(y_true.min(), y_pred.min()) - 0.5
    hi = max(y_true.max(), y_pred.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="Perfect prediction")
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    mae = np.mean(np.abs(y_true - y_pred))
    ax.annotate(f"R² = {r2:.4f}\nMAE = {mae:.4f}%",
                xy=(0.05, 0.88), xycoords="axes fraction", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8))
    ax.set_xlabel("Actual CSEE Pass Rate (%)")
    ax.set_ylabel("Predicted CSEE Pass Rate (%)")
    ax.set_title(f"Actual vs. Predicted — {model_name}")
    ax.legend()

    # ---- Right: sorted comparison ----
    ax2 = axes[1]
    idx = np.argsort(y_true)
    ax2.plot(range(len(y_true)), y_true[idx], "o-", color="steelblue",
             ms=4, alpha=0.8, label="Actual")
    ax2.plot(range(len(y_pred)), y_pred[idx], "s--", color="crimson",
             ms=4, alpha=0.8, label="Predicted")
    ax2.set_xlabel("Observation (sorted by actual)")
    ax2.set_ylabel("CSEE Pass Rate (%)")
    ax2.set_title("Sorted Actual vs. Predicted")
    ax2.legend()

    plt.suptitle(f"Prediction Evaluation — {model_name}", fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def residual_diagnostics_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Four-panel residual diagnostic plot."""
    residuals = y_true - y_pred
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, s=40, alpha=0.7, color="steelblue",
                       edgecolors="white", lw=0.4)
    axes[0, 0].axhline(0, color="black", lw=1.5, linestyle="--")
    axes[0, 0].set_xlabel("Fitted Values")
    axes[0, 0].set_ylabel("Residuals (%)")
    axes[0, 0].set_title("Residuals vs. Fitted")

    # Q-Q plot
    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(residuals, dist="norm")
    axes[0, 1].scatter(osm, osr, s=35, alpha=0.7, color="darkorange", edgecolors="white")
    axes[0, 1].plot(osm, slope * np.array(osm) + intercept, "r-", lw=1.5)
    axes[0, 1].set_xlabel("Theoretical Quantiles")
    axes[0, 1].set_ylabel("Sample Quantiles")
    axes[0, 1].set_title("Normal Q-Q Plot")

    # Residual distribution
    axes[1, 0].hist(residuals, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
    axes[1, 0].axvline(0, color="black", lw=1.5, linestyle="--")
    axes[1, 0].axvline(residuals.mean(), color="crimson", lw=1.5,
                       linestyle=":", label=f"Mean = {residuals.mean():.3f}")
    axes[1, 0].set_xlabel("Residual (%)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Residual Distribution")
    axes[1, 0].legend()

    # Scale-location
    sqrt_abs_res = np.sqrt(np.abs(residuals))
    axes[1, 1].scatter(y_pred, sqrt_abs_res, s=40, alpha=0.7, color="mediumseagreen",
                       edgecolors="white", lw=0.4)
    axes[1, 1].set_xlabel("Fitted Values")
    axes[1, 1].set_ylabel("sqrt(|Residuals|)")
    axes[1, 1].set_title("Scale-Location Plot")
    # Smooth trend
    sort_idx = np.argsort(y_pred)
    smooth = pd.Series(sqrt_abs_res[sort_idx]).rolling(5, min_periods=1).mean()
    axes[1, 1].plot(y_pred[sort_idx], smooth, "r-", lw=1.5, alpha=0.8)

    plt.suptitle(f"Residual Diagnostics — {model_name}", fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def learning_curve_plot(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "Model",
    cv: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot learning curves to diagnose bias/variance."""
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.2, 1.0, 8), n_jobs=-1,
    )
    train_mean = -train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = -val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Training MAE")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color="steelblue")
    ax.plot(train_sizes, val_mean, "s-", color="darkorange", label="CV Validation MAE")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.2, color="darkorange")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Mean Absolute Error (%)")
    ax.set_title(f"Learning Curve — {model_name}", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def model_comparison_bar_chart(
    comparison_df: pd.DataFrame,
    metric: str = "CV_MAE_mean",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing models on a given metric."""
    df = comparison_df.copy().sort_values(metric, ascending=(metric != "CV_R2_mean"))
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.45)))
    colours = ["crimson" if i == 0 else "steelblue" for i in range(len(df))]
    bars = ax.barh(df["Model"], df[metric], color=colours, alpha=0.85)
    # Annotate best
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(2)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ')}", fontweight="bold")
    ax.invert_yaxis()
    # Value labels
    for bar, val in zip(bars, df[metric]):
        ax.text(val + 0.001 * abs(val), bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def error_distribution_by_year(
    df_test: pd.DataFrame,
    y_pred: np.ndarray,
    target_col: str = "csee_pass_rate",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Show prediction errors grouped by year."""
    df_plot = df_test[["year", target_col]].copy()
    df_plot["predicted"] = y_pred
    df_plot["error"]     = df_plot[target_col] - df_plot["predicted"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    years = sorted(df_plot["year"].unique())

    # Box plot by year
    data = [df_plot[df_plot["year"] == yr]["error"].values for yr in years]
    bp = axes[0].boxplot(data, labels=years, patch_artist=True,
                         medianprops=dict(color="black", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    axes[0].axhline(0, color="crimson", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Prediction Error (%)")
    axes[0].set_title("Prediction Error Distribution by Year")

    # Mean error by year
    yr_err = df_plot.groupby("year")["error"].agg(["mean", "std"]).reset_index()
    axes[1].bar(yr_err["year"], yr_err["mean"], yerr=yr_err["std"],
                color="steelblue", alpha=0.8, capsize=5, width=0.5)
    axes[1].axhline(0, color="crimson", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Mean Error (%)")
    axes[1].set_title("Mean Prediction Error by Year (with ±1 SD)")

    plt.suptitle("Temporal Error Analysis", fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
