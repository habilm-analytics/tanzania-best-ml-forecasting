"""
visualization.py
================
Publication-quality visualisation library for Tanzania BEST education data.
Produces 50+ chart types covering enrolment trends, teacher supply,
infrastructure, examination performance, gender equity, and clustering.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

logger = logging.getLogger("visualization")

EXPECTED_REGIONS = [
    "ARUSHA","DAR ES SALAAM","DODOMA","GEITA","IRINGA","KAGERA",
    "KATAVI","KIGOMA","KILIMANJARO","LINDI","MANYARA","MARA",
    "MBEYA","MOROGORO","MTWARA","MWANZA","NJOMBE","PWANI",
    "RUKWA","RUVUMA","SHINYANGA","SIMIYU","SINGIDA","SONGWE",
    "TABORA","TANGA",
]

# Global style
STYLE = {
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
plt.rcParams.update(STYLE)
sns.set_palette("tab10")


def _save(fig: plt.Figure, path: Optional[str]) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")


# ============================================================
# 1. CSEE Pass Rate Visualisations
# ============================================================

def plot_csee_trend(csee: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
    """National CSEE pass rate trend with annotated values."""
    df = csee[csee["year"] >= 2011].sort_values("year")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(df["year"], df["csee_pass_rate"], alpha=0.18, color="steelblue")
    ax.plot(df["year"], df["csee_pass_rate"], "o-", color="steelblue", lw=2.5, ms=8)
    for _, r in df.iterrows():
        ax.annotate(f"{r.csee_pass_rate:.1f}%", (r.year, r.csee_pass_rate),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_ylim(35, 98)
    ax.set_xlabel("Year")
    ax.set_ylabel("CSEE Pass Rate (%)")
    ax.set_title("National CSEE Pass Rate Trend (2011-2023)\n"
                 "Certificate of Secondary Education Examination", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, save_path)
    return fig


def plot_csee_division_breakdown(csee: pd.DataFrame,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """Stacked area: Div I, II, III, IV and fail over time."""
    df = csee[csee["year"] >= 2015].sort_values("year")
    if "csee_div1_pct" not in df.columns:
        return plt.figure()
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["year"], df["csee_div1_pct"], "o-", label="Division I", color="#2ecc71", lw=2)
    ax.plot(df["year"], df["csee_pass_rate"], "s--", label="Total Pass Rate", color="steelblue", lw=2)
    ax.fill_between(df["year"], df["csee_div1_pct"], alpha=0.2, color="#2ecc71")
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("CSEE Performance: Division I vs Total Pass Rate", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, save_path)
    return fig


def plot_csee_candidates_trend(csee: pd.DataFrame,
                                save_path: Optional[str] = None) -> plt.Figure:
    """Dual-axis: candidates examined and pass rate."""
    df = csee[csee["year"] >= 2015].dropna(subset=["csee_candidates"]).sort_values("year")
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()
    ax1.bar(df["year"], df["csee_candidates"] / 1e3, color="steelblue",
            alpha=0.6, width=0.5, label="Candidates (000s)")
    ax2.plot(df["year"], df["csee_pass_rate"], "o-", color="crimson",
             lw=2.5, ms=8, label="Pass Rate %")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Candidates Examined (000s)", color="steelblue")
    ax2.set_ylabel("Pass Rate (%)", color="crimson")
    ax1.tick_params(axis="y", colors="steelblue")
    ax2.tick_params(axis="y", colors="crimson")
    ax1.set_title("CSEE Candidates Examined vs National Pass Rate", fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    _save(fig, save_path)
    return fig


# ============================================================
# 2. Enrolment Visualisations
# ============================================================

def plot_enrolment_trend(panel: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
    """National total enrolment trend with govt/non-govt split."""
    yr = panel.groupby("year").agg(
        total=("enrolment_f1f4", "sum"),
        govt=("govt_schools", "sum"),
        nongovt=("nongovt_schools", "sum"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].fill_between(yr["year"], yr["total"] / 1e6, alpha=0.25, color="steelblue")
    axes[0].plot(yr["year"], yr["total"] / 1e6, "o-", color="steelblue", lw=2.5, ms=8)
    for _, r in yr.iterrows():
        axes[0].annotate(f"{r.total/1e6:.2f}M", (r.year, r.total/1e6),
                         textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Enrolment (millions)")
    axes[0].set_title("Total Secondary Enrolment (Form 1-4)", fontweight="bold")

    width = 0.35
    x = np.arange(len(yr))
    axes[1].bar(x - width/2, yr["govt"], width, label="Government", color="steelblue", alpha=0.8)
    axes[1].bar(x + width/2, yr["nongovt"], width, label="Non-Government", color="darkorange", alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(yr["year"])
    axes[1].set_ylabel("Number of Schools")
    axes[1].set_title("Government vs Non-Government Schools", fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_enrolment_by_region(panel: pd.DataFrame,
                              year: int = 2023,
                              save_path: Optional[str] = None) -> plt.Figure:
    """Horizontal bar chart: enrolment by region for a given year."""
    df = panel[panel["year"] == year].dropna(subset=["enrolment_f1f4"])
    df = df.sort_values("enrolment_f1f4", ascending=False)
    median_e = df["enrolment_f1f4"].median()
    fig, ax = plt.subplots(figsize=(12, 8))
    colours = ["steelblue" if v > median_e else "salmon" for v in df["enrolment_f1f4"]]
    ax.barh(df["region"], df["enrolment_f1f4"], color=colours, alpha=0.85)
    ax.axvline(median_e, color="gray", lw=1.5, linestyle="--", label="Median")
    ax.set_xlabel("Total Enrolment (Form 1-4)")
    ax.set_title(f"Secondary School Enrolment by Region ({year})", fontweight="bold")
    ax.invert_yaxis()
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_enrolment_growth_heatmap(panel: pd.DataFrame,
                                   save_path: Optional[str] = None) -> plt.Figure:
    """Heatmap of enrolment by region and year."""
    pivot = panel.pivot_table(index="region", columns="year",
                               values="enrolment_f1f4", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.heatmap(pivot / 1e3, ax=ax, cmap="Blues", annot=True, fmt=".0f",
                linewidths=0.4, cbar_kws={"label": "Enrolment (000s)"})
    ax.set_title("Secondary Enrolment by Region and Year (000s)", fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# 3. Teacher and PTR Visualisations
# ============================================================

def plot_teacher_trend(panel: pd.DataFrame,
                        csee: Optional[pd.DataFrame] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
    """Teacher count trend with optional pass rate overlay."""
    yr_t = panel.groupby("year")["total_teachers"].sum().reset_index()
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(yr_t["year"], yr_t["total_teachers"] / 1e3, color="darkorange",
            alpha=0.7, width=0.5, label="Total Teachers (000s)")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Total Teachers (000s)", color="darkorange")
    ax1.tick_params(axis="y", colors="darkorange")
    if csee is not None:
        ax2 = ax1.twinx()
        df_c = csee[csee["year"].isin(yr_t["year"])].sort_values("year")
        ax2.plot(df_c["year"], df_c["csee_pass_rate"], "o-",
                 color="steelblue", lw=2.5, ms=8, label="CSEE Pass Rate %")
        ax2.set_ylabel("CSEE Pass Rate (%)", color="steelblue")
        ax2.tick_params(axis="y", colors="steelblue")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("National Teacher Count vs CSEE Pass Rate", fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_ptr_by_region(panel: pd.DataFrame,
                        save_path: Optional[str] = None) -> plt.Figure:
    """Horizontal bar: average PTR by region with threshold line."""
    reg_ptr = (panel.groupby("region")["ptr_regional"]
               .mean().sort_values(ascending=False).reset_index())
    reg_ptr.columns = ["region", "avg_ptr"]
    fig, ax = plt.subplots(figsize=(12, 8))
    colours = ["crimson" if v > 30 else "steelblue" for v in reg_ptr["avg_ptr"]]
    ax.barh(reg_ptr["region"], reg_ptr["avg_ptr"], color=colours, alpha=0.85)
    ax.axvline(30, color="darkred", lw=2, linestyle="--", label="30:1 threshold")
    ax.set_xlabel("Average Pupil-Teacher Ratio")
    ax.set_title("Average Regional Pupil-Teacher Ratio (2020-2024)\n"
                 "Red bars exceed the 30:1 quality threshold", fontweight="bold")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_ptr_trend(panel: pd.DataFrame, csee: pd.DataFrame,
                   save_path: Optional[str] = None) -> plt.Figure:
    """Dual-axis: national PTR trend vs CSEE pass rate."""
    ts = (panel.groupby("year")
          .agg(ptr=("ptr_national", "first"),
               pass_rate=("csee_pass_rate", "first"))
          .reset_index().dropna())
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()
    l1, = ax1.plot(ts["year"], ts["ptr"], "s-", color="darkorange",
                   lw=2.5, ms=8, label="National PTR")
    l2, = ax2.plot(ts["year"], ts["pass_rate"], "o-", color="steelblue",
                   lw=2.5, ms=8, label="CSEE Pass Rate %")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Pupil-Teacher Ratio", color="darkorange")
    ax2.set_ylabel("CSEE Pass Rate (%)", color="steelblue")
    ax1.tick_params(axis="y", colors="darkorange")
    ax2.tick_params(axis="y", colors="steelblue")
    ax1.set_title("National PTR vs CSEE Pass Rate Trend", fontweight="bold")
    ax1.legend(handles=[l1, l2], loc="lower right")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_qualified_teacher_ratio(panel: pd.DataFrame,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """Trend in qualified teacher ratio with annotation."""
    yr = panel.groupby("year")["qualified_teacher_ratio"].first().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(yr["year"], yr["qualified_teacher_ratio"] * 100,
                    alpha=0.2, color="mediumseagreen")
    ax.plot(yr["year"], yr["qualified_teacher_ratio"] * 100, "o-",
            color="mediumseagreen", lw=2.5, ms=8)
    for _, r in yr.iterrows():
        if pd.notna(r.qualified_teacher_ratio):
            ax.annotate(f"{r.qualified_teacher_ratio*100:.1f}%",
                        (r.year, r.qualified_teacher_ratio * 100),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(85, 105)
    ax.set_xlabel("Year")
    ax.set_ylabel("Qualified Teachers (%)")
    ax.set_title("National Qualified Teacher Ratio (2020-2024)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# 4. Infrastructure Visualisations
# ============================================================

def plot_electricity_by_region(panel: pd.DataFrame,
                                save_path: Optional[str] = None) -> plt.Figure:
    """Horizontal bar: average electricity access by region."""
    reg_e = (panel.groupby("region")["pct_schools_electricity"]
             .mean().sort_values().reset_index())
    reg_e.columns = ["region", "avg_elec"]
    fig, ax = plt.subplots(figsize=(12, 8))
    colours = ["mediumseagreen" if v >= 70 else "darkorange"
               for v in reg_e["avg_elec"]]
    ax.barh(reg_e["region"], reg_e["avg_elec"], color=colours, alpha=0.85)
    ax.axvline(70, color="green", lw=2, linestyle="--", label="70% benchmark")
    ax.set_xlabel("% of Schools with Electricity Access")
    ax.set_title("Secondary School Electricity Access by Region\n"
                 "(Average 2021-2024; green = meets 70% benchmark)", fontweight="bold")
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_electricity_trend(panel: pd.DataFrame,
                            save_path: Optional[str] = None) -> plt.Figure:
    """Boxplot of electricity access distribution across years."""
    fig, ax = plt.subplots(figsize=(10, 5))
    years = sorted(panel["year"].unique())
    data = [panel[panel["year"] == yr]["pct_schools_electricity"].dropna().values
            for yr in years]
    bp = ax.boxplot(data, labels=years, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("% Schools with Electricity")
    ax.set_title("Distribution of School Electricity Access Across Years", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_ict_by_region(panel: pd.DataFrame,
                        save_path: Optional[str] = None) -> plt.Figure:
    """Horizontal bar: desktop computers per school by region."""
    reg_ict = (panel.groupby("region")["desktops_per_school"]
               .mean().sort_values(ascending=False).reset_index())
    reg_ict.columns = ["region", "avg_desktops"]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(reg_ict["region"], reg_ict["avg_desktops"],
            color="mediumpurple", alpha=0.85)
    ax.set_xlabel("Average Desktop Computers per School")
    ax.set_title("ICT Penetration by Region — Desktops per School\n"
                 "(Average 2021-2024)", fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_infrastructure_scatter(panel: pd.DataFrame,
                                 year: int = 2023,
                                 save_path: Optional[str] = None) -> plt.Figure:
    """Scatter: electricity access vs ICT penetration coloured by enrolment."""
    df = panel[panel["year"] == year].dropna(
        subset=["pct_schools_electricity", "desktops_per_school"])
    fig, ax = plt.subplots(figsize=(11, 7))
    sc = ax.scatter(df["pct_schools_electricity"], df["desktops_per_school"],
                    c=df["enrolment_f1f4"], cmap="viridis",
                    s=100, alpha=0.85, edgecolors="white", lw=0.5)
    for _, row in df.iterrows():
        ax.annotate(row["region"],
                    (row["pct_schools_electricity"], row["desktops_per_school"]),
                    fontsize=6.5, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="Total Enrolment (Form 1-4)")
    ax.set_xlabel("% Schools with Electricity (%)")
    ax.set_ylabel("Desktop Computers per School")
    ax.set_title(f"Infrastructure: Electricity vs ICT by Region ({year})", fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# 5. Dropout & Completion
# ============================================================

def plot_dropout_completion(panel: pd.DataFrame,
                             save_path: Optional[str] = None) -> plt.Figure:
    """Dual-line: national dropout and completion rate trends."""
    ts = (panel.groupby("year")
          .agg(dropout=("dropout_rate_pct", "first"),
               completion=("gross_completion_rate", "first"))
          .reset_index().dropna())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts["year"], ts["dropout"], "D-", color="crimson",
            lw=2.5, ms=8, label="Dropout Rate (%)")
    ax.plot(ts["year"], ts["completion"], "o-", color="teal",
            lw=2.5, ms=8, label="Gross Completion Rate (%)")
    for _, r in ts.iterrows():
        if pd.notna(r.dropout):
            ax.annotate(f"{r.dropout:.1f}%", (r.year, r.dropout),
                        textcoords="offset points", xytext=(0, -16),
                        ha="center", fontsize=8, color="crimson")
        if pd.notna(r.completion):
            ax.annotate(f"{r.completion:.1f}%", (r.year, r.completion),
                        textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color="teal")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rate (%)")
    ax.set_title("National Secondary School Dropout vs Completion Rate", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_dropout_by_region(panel: pd.DataFrame,
                            save_path: Optional[str] = None) -> plt.Figure:
    """Violin plot of regional dropout rates across years."""
    df = panel.dropna(subset=["dropout_rate_regional"])
    fig, ax = plt.subplots(figsize=(12, 6))
    years = sorted(df["year"].unique())
    data = [df[df["year"] == yr]["dropout_rate_regional"].values for yr in years]
    vp = ax.violinplot(data, positions=range(len(years)), showmedians=True,
                       showextrema=True)
    for body in vp["bodies"]:
        body.set_facecolor("steelblue")
        body.set_alpha(0.6)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_xlabel("Year")
    ax.set_ylabel("Regional Dropout Rate (%)")
    ax.set_title("Distribution of Regional Dropout Rates by Year", fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# 6. Correlation & Bivariate
# ============================================================

def plot_correlation_heatmap(panel: pd.DataFrame,
                              cols: Optional[List[str]] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
    """Full correlation heatmap of key education indicators."""
    if cols is None:
        cols = [c for c in [
            "csee_pass_rate", "total_schools", "enrolment_f1f4", "total_teachers",
            "pct_schools_electricity", "desktops_per_school", "ptr_regional",
            "ptr_national", "qualified_teacher_ratio", "dropout_rate_pct",
            "gross_completion_rate", "teachers_per_school", "nongovt_share",
            "infra_index", "education_quality_index",
        ] if c in panel.columns]
    corr = panel[cols].dropna().corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.zeros_like(corr, dtype=bool)
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.4, annot_kws={"size": 7.5},
                square=True, vmin=-1, vmax=1, mask=mask)
    ax.set_title("Correlation Matrix — Tanzania Education System Indicators",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_pairplot(panel: pd.DataFrame,
                  cols: Optional[List[str]] = None,
                  save_path: Optional[str] = None) -> plt.Figure:
    """Pairplot of key predictors and target."""
    if cols is None:
        cols = [c for c in [
            "csee_pass_rate", "ptr_national", "pct_schools_electricity",
            "desktops_per_school", "qualified_teacher_ratio", "dropout_rate_pct",
        ] if c in panel.columns]
    df = panel[cols].dropna()
    g = sns.pairplot(df, diag_kind="kde",
                     plot_kws={"alpha": 0.5, "color": "steelblue", "s": 22})
    g.figure.suptitle("Pairwise Relationships: Key Predictors vs CSEE Pass Rate",
                       y=1.01, fontsize=12, fontweight="bold")
    if save_path:
        g.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return g.figure


def plot_scatter_ptr_passrate(panel: pd.DataFrame,
                               save_path: Optional[str] = None) -> plt.Figure:
    """Scatter: PTR vs pass rate coloured by year."""
    df = panel.dropna(subset=["ptr_regional", "csee_pass_rate"])
    fig, ax = plt.subplots(figsize=(10, 6))
    years = sorted(df["year"].unique())
    palette = plt.cm.viridis(np.linspace(0, 1, len(years)))
    for i, yr in enumerate(years):
        sub = df[df["year"] == yr]
        ax.scatter(sub["ptr_regional"], sub["csee_pass_rate"],
                   color=palette[i], alpha=0.7, s=55, label=str(yr),
                   edgecolors="white", lw=0.4)
    ax.set_xlabel("Regional Pupil-Teacher Ratio")
    ax.set_ylabel("CSEE Pass Rate (%)")
    ax.set_title("PTR vs CSEE Pass Rate by Region and Year", fontweight="bold")
    ax.legend(title="Year")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_electricity_vs_passrate(panel: pd.DataFrame,
                                  save_path: Optional[str] = None) -> plt.Figure:
    """Scatter: electricity access vs CSEE pass rate."""
    df = panel.dropna(subset=["pct_schools_electricity", "csee_pass_rate"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(df["pct_schools_electricity"], df["csee_pass_rate"],
               c=df["year"], cmap="plasma", s=60, alpha=0.75,
               edgecolors="white", lw=0.4)
    # Regression line
    x = df["pct_schools_electricity"].values
    y = df["csee_pass_rate"].values
    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m * xline + b, "r--", lw=1.8, label=f"Trend (slope={m:.3f})")
    ax.set_xlabel("% Schools with Electricity")
    ax.set_ylabel("CSEE Pass Rate (%)")
    ax.set_title("Electricity Access vs CSEE Pass Rate", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# 7. Distribution Plots
# ============================================================

def plot_distributions(panel: pd.DataFrame,
                        save_path: Optional[str] = None) -> plt.Figure:
    """KDE + histogram for key numeric indicators."""
    cols = [c for c in [
        "csee_pass_rate", "ptr_regional", "pct_schools_electricity",
        "desktops_per_school", "dropout_rate_pct", "gross_completion_rate",
    ] if c in panel.columns]
    n = len(cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        data = panel[col].dropna()
        axes[i].hist(data, bins=15, density=True, color="steelblue",
                     alpha=0.5, edgecolor="white")
        data.plot.kde(ax=axes[i], color="crimson", lw=2)
        axes[i].axvline(data.mean(), color="darkred", lw=1.5, linestyle="--",
                        label=f"Mean={data.mean():.1f}")
        axes[i].set_xlabel(col.replace("_", " ").title())
        axes[i].set_ylabel("Density")
        axes[i].set_title(col.replace("_", " ").title())
        axes[i].legend(fontsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribution of Key Education Indicators", fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_boxplots_by_year(panel: pd.DataFrame,
                           save_path: Optional[str] = None) -> plt.Figure:
    """Boxplots of key indicators across years."""
    cols = [c for c in [
        "pct_schools_electricity", "ptr_regional",
        "desktops_per_school", "dropout_rate_regional",
    ] if c in panel.columns]
    years = sorted(panel["year"].unique())
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]
    colours = ["steelblue", "darkorange", "mediumseagreen", "mediumpurple"]
    for ax, col, colour in zip(axes, cols, colours):
        data = [panel[panel["year"] == yr][col].dropna().values for yr in years]
        bp = ax.boxplot(data, labels=years, patch_artist=True,
                        medianprops=dict(color="black", lw=2))
        for patch in bp["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)
        ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=10)
        ax.set_xlabel("Year")
    plt.suptitle("Regional Indicator Distribution by Year", fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# 8. PCA and Clustering
# ============================================================

def plot_pca_biplot(panel: pd.DataFrame,
                    save_path: Optional[str] = None) -> plt.Figure:
    """2D PCA biplot of regional education profiles."""
    feat_cols = [c for c in [
        "pct_schools_electricity", "desktops_per_school", "ptr_regional",
        "teachers_per_school", "nongovt_share", "dropout_rate_regional",
        "enrolment_per_school",
    ] if c in panel.columns]
    df = panel.dropna(subset=feat_cols + ["region"])
    if len(df) < 10:
        return plt.figure()

    X = StandardScaler().fit_transform(df[feat_cols])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(11, 8))
    years = sorted(df["year"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(years)))
    for i, yr in enumerate(years):
        mask = df["year"] == yr
        ax.scatter(coords[mask, 0], coords[mask, 1], color=palette[i],
                   alpha=0.65, s=60, label=str(yr), edgecolors="white", lw=0.4)

    # Annotate a subset of regions
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 5 == 0:
            ax.annotate(row["region"], (coords[idx, 0], coords[idx, 1]),
                        fontsize=6, alpha=0.7)

    # Loadings
    scale = 3
    for j, feat in enumerate(feat_cols):
        ax.arrow(0, 0, pca.components_[0, j] * scale, pca.components_[1, j] * scale,
                 head_width=0.08, head_length=0.08, fc="crimson", ec="crimson", alpha=0.55)
        ax.text(pca.components_[0, j] * scale * 1.15,
                pca.components_[1, j] * scale * 1.15,
                feat.replace("_", " "), fontsize=7, color="crimson")

    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% variance)")
    ax.set_title("PCA Biplot — Regional Education Profiles", fontweight="bold")
    ax.legend(title="Year", loc="upper right")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


def plot_kmeans_clusters(panel: pd.DataFrame, n_clusters: int = 4,
                          save_path: Optional[str] = None) -> plt.Figure:
    """K-means clustering of regions by education profile."""
    feat_cols = [c for c in [
        "pct_schools_electricity", "desktops_per_school", "ptr_regional",
        "teachers_per_school", "nongovt_share",
    ] if c in panel.columns]
    df = panel[panel["year"] == panel["year"].max()].dropna(subset=feat_cols).copy()
    if len(df) < n_clusters + 1:
        return plt.figure()

    X = StandardScaler().fit_transform(df[feat_cols])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colours = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    # Left: scatter in PCA space
    for cl in range(n_clusters):
        mask = df["cluster"] == cl
        axes[0].scatter(coords[mask, 0], coords[mask, 1],
                        color=colours[cl], s=80, alpha=0.85,
                        label=f"Cluster {cl+1}", edgecolors="white")
    for idx, (_, row) in enumerate(df.iterrows()):
        axes[0].annotate(row["region"], (coords[idx, 0], coords[idx, 1]),
                         fontsize=6.5, alpha=0.8)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].set_title("K-Means Clusters in PCA Space", fontweight="bold")
    axes[0].legend()

    # Right: bar chart with cluster assignment
    df_sorted = df.sort_values("cluster")
    bar_colours = [colours[c] for c in df_sorted["cluster"]]
    axes[1].barh(df_sorted["region"], df_sorted["cluster"] + 1,
                 color=bar_colours, alpha=0.85)
    axes[1].set_xlabel("Cluster Assignment")
    axes[1].set_title("Region Cluster Assignments", fontweight="bold")
    axes[1].invert_yaxis()

    plt.suptitle(f"K-Means Clustering of Tanzania Regions (k={n_clusters})",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# 9. Missing Data & Quality
# ============================================================

def plot_missing_heatmap(panel: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Missing data heatmap."""
    num_cols = panel.select_dtypes(include=[float, int]).columns.tolist()
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(panel[num_cols].isnull().T.astype(int), ax=ax, cbar=False,
                cmap="Reds", xticklabels=False, linewidths=0.2)
    ax.set_title("Missing Data Map — Red = Missing Value", fontweight="bold")
    ax.set_xlabel("Panel Observations (rows)")
    plt.tight_layout()
    _save(fig, save_path)
    return fig


# ============================================================
# 10. Schools count
# ============================================================

def plot_schools_by_type(panel: pd.DataFrame,
                          save_path: Optional[str] = None) -> plt.Figure:
    """Stacked bar: govt vs non-govt schools by year."""
    yr = panel.groupby("year").agg(
        govt=("govt_schools", "sum"),
        nongovt=("nongovt_schools", "sum"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(yr))
    ax.bar(x, yr["govt"], label="Government", color="steelblue", alpha=0.85)
    ax.bar(x, yr["nongovt"], bottom=yr["govt"], label="Non-Government",
           color="darkorange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(yr["year"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Schools")
    ax.set_title("Secondary Schools by Ownership Type (National Total)", fontweight="bold")
    ax.legend()
    fmt = lambda x, _: f"{x/1e3:.1f}K"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt))
    plt.tight_layout()
    _save(fig, save_path)
    return fig
