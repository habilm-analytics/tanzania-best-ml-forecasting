"""
streamlit_app.py
================
Interactive Streamlit dashboard for Tanzania BEST Education ML Forecasting.
Allows users to explore data, compare regions, generate predictions,
and download reports.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)

Usage:
    streamlit run app/streamlit_app.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit not installed. Run: pip install streamlit")
    sys.exit(0)

from utilities import ProjectPaths, load_model
from feature_engineering import get_model_features

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tanzania BEST Education ML Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

PATHS = ProjectPaths()

EXPECTED_REGIONS = [
    "ARUSHA", "DAR ES SALAAM", "DODOMA", "GEITA", "IRINGA", "KAGERA",
    "KATAVI", "KIGOMA", "KILIMANJARO", "LINDI", "MANYARA", "MARA",
    "MBEYA", "MOROGORO", "MTWARA", "MWANZA", "NJOMBE", "PWANI",
    "RUKWA", "RUVUMA", "SHINYANGA", "SIMIYU", "SINGIDA", "SONGWE",
    "TABORA", "TANGA",
]


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load the processed panel and CSEE series."""
    try:
        panel = pd.read_csv(PATHS.processed("best_panel_features.csv"))
        csee  = pd.read_csv(PATHS.processed("csee_national_trend.csv"))
        model_feats = list(pd.read_csv(PATHS.table("model_features.csv"))["feature"])
        return panel, csee, model_feats
    except FileNotFoundError:
        return None, None, None


@st.cache_resource
def load_ml_model():
    """Load the trained model and scaler."""
    try:
        model  = load_model(PATHS.model("gradient_boosting.pkl"))
        scaler = load_model(PATHS.model("feature_scaler.pkl"))
        return model, scaler
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Flag_of_Tanzania.svg/320px-Flag_of_Tanzania.svg.png",
                 width=160)
st.sidebar.title("BEST Education ML")
st.sidebar.markdown("**Tanzania BEST Data (2020-2024)**")
st.sidebar.markdown("*Prepared by Habil Masawika*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Regional Analysis", "Prediction Engine",
     "Forecast (2025-2030)", "Data Explorer", "Model Information"],
    index=0,
)

panel, csee_nat, model_feats = load_data()
model, scaler = load_ml_model()

if panel is None:
    st.error("Data not found. Please run notebooks 01-05 first to generate processed data.")
    st.stop()

# ---------------------------------------------------------------------------
# Helper: small matplotlib figure in Streamlit
# ---------------------------------------------------------------------------
def render_fig(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ===========================================================================
# PAGE 1: Overview
# ===========================================================================
if page == "Overview":
    st.title("📊 Tanzania Secondary Education — National Overview")
    st.markdown("""
    This dashboard presents key secondary education indicators from the Tanzania
    **Basic Education Statistics (BEST)** annual reports (2020-2024), integrated
    with a machine learning model trained to predict CSEE examination pass rates.
    """)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    latest = panel[panel["year"] == panel["year"].max()]
    col1.metric("Latest CSEE Pass Rate",
                f"{panel['csee_pass_rate'].dropna().iloc[-1]:.1f}%",
                delta=f"+{panel.groupby('year')['csee_pass_rate'].first().diff().iloc[-1]:.1f}pp")
    col2.metric("Total Schools (latest)",
                f"{int(latest['total_schools'].sum()):,}")
    col3.metric("Total Enrolment (latest)",
                f"{int(latest['enrolment_f1f4'].sum()/1e6*10)/10:.1f}M")
    col4.metric("National PTR (latest)",
                f"{panel.groupby('year')['ptr_national'].first().iloc[-1]:.0f}:1")

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("CSEE Pass Rate Trend")
        df_csee = csee_nat[csee_nat["year"] >= 2015].sort_values("year")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(df_csee["year"], df_csee["csee_pass_rate"], alpha=0.2, color="steelblue")
        ax.plot(df_csee["year"], df_csee["csee_pass_rate"], "o-", color="steelblue", lw=2.5, ms=8)
        for _, r in df_csee.iterrows():
            ax.annotate(f"{r.csee_pass_rate:.1f}%", (r.year, r.csee_pass_rate),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
        ax.set_ylim(55, 98)
        ax.set_xlabel("Year")
        ax.set_ylabel("Pass Rate (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        render_fig(fig)

    with col_r:
        st.subheader("Enrolment vs Teacher Growth")
        yr_data = panel.groupby("year").agg(
            enrolment=("enrolment_f1f4", "sum"),
            teachers=("total_teachers", "sum"),
        ).reset_index()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax2 = ax.twinx()
        ax.bar(yr_data["year"], yr_data["enrolment"]/1e6, color="steelblue",
               alpha=0.6, width=0.35, label="Enrolment (M)")
        ax2.plot(yr_data["year"], yr_data["teachers"]/1e3, "s-",
                 color="darkorange", lw=2.5, ms=8, label="Teachers (000s)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Enrolment (millions)", color="steelblue")
        ax2.set_ylabel("Teachers (000s)", color="darkorange")
        ax.spines["top"].set_visible(False)
        render_fig(fig)

    # Electricity access map (bar chart)
    st.subheader("Electricity Access by Region (Average 2021-2024)")
    reg_e = panel.groupby("region")["pct_schools_electricity"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(14, 6))
    colours_e = ["mediumseagreen" if v >= 70 else "darkorange" for v in reg_e.values]
    ax.barh(reg_e.index, reg_e.values, color=colours_e, alpha=0.85)
    ax.axvline(70, color="green", lw=1.8, linestyle="--", label="70% benchmark")
    ax.set_xlabel("% Schools with Electricity")
    ax.legend()
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    render_fig(fig)


# ===========================================================================
# PAGE 2: Regional Analysis
# ===========================================================================
elif page == "Regional Analysis":
    st.title("🗺️ Regional Deep Dive")

    region = st.selectbox("Select Region", sorted(EXPECTED_REGIONS), index=0)
    df_reg = panel[panel["region"] == region].sort_values("year")

    if df_reg.empty:
        st.warning(f"No data found for {region}.")
        st.stop()

    st.markdown(f"### {region} — Education Profile (2020-2024)")

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Enrolment", f"{df_reg['enrolment_f1f4'].mean():,.0f}")
    col2.metric("Avg PTR",       f"{df_reg['ptr_regional'].mean():.1f}:1")
    col3.metric("Avg Electricity",
                f"{df_reg['pct_schools_electricity'].mean():.1f}%"
                if "pct_schools_electricity" in df_reg.columns else "N/A")

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Enrolment Trend")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(df_reg["year"], df_reg["enrolment_f1f4"]/1e3, "o-",
                color="steelblue", lw=2.5, ms=8)
        ax.set_xlabel("Year")
        ax.set_ylabel("Enrolment (000s)")
        ax.grid(axis="y", alpha=0.3)
        render_fig(fig)

    with col_r:
        st.subheader("PTR Trend")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(df_reg["year"], df_reg["ptr_regional"], "o-",
                color="darkorange", lw=2.5, ms=8)
        ax.axhline(30, color="red", lw=1.5, linestyle="--", label="30:1 threshold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Pupil-Teacher Ratio")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        render_fig(fig)

    # Full indicator table
    st.subheader("Full Indicator Table")
    display_cols = [c for c in [
        "year", "total_schools", "enrolment_f1f4", "total_teachers",
        "ptr_regional", "pct_schools_electricity", "desktops_per_school",
        "dropout_rate_regional", "csee_pass_rate",
    ] if c in df_reg.columns]
    st.dataframe(df_reg[display_cols].round(2), use_container_width=True)

    # Download
    csv = df_reg.to_csv(index=False)
    st.download_button(f"Download {region} Data (CSV)", csv,
                       file_name=f"{region.lower().replace(' ','_')}_profile.csv")


# ===========================================================================
# PAGE 3: Prediction Engine
# ===========================================================================
elif page == "Prediction Engine":
    st.title("🤖 ML Prediction Engine")
    st.markdown("""
    Enter education system inputs to generate a predicted CSEE pass rate.
    The model is a tuned **Gradient Boosting Regressor** trained on 2020-2023 data.
    """)

    if model is None or scaler is None:
        st.error("Trained model not found. Run Notebook 05 first.")
        st.stop()

    st.subheader("Input Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        ptr_nat           = st.slider("National PTR", 15.0, 40.0,
                                       float(panel["ptr_national"].mean()), 0.5)
        qual_teach_ratio  = st.slider("Qualified Teacher Ratio", 0.90, 1.00,
                                       float(panel["qualified_teacher_ratio"].mean()), 0.01)
        dropout_rate      = st.slider("Dropout Rate (%)", 0.0, 15.0,
                                       float(panel["dropout_rate_pct"].mean()), 0.1)

    with col2:
        completion_rate   = st.slider("Gross Completion Rate (%)", 20.0, 80.0,
                                       float(panel["gross_completion_rate"].mean()), 0.5)
        pct_electricity   = st.slider("% Schools with Electricity", 10.0, 100.0,
                                       float(panel["pct_schools_electricity"].mean()), 1.0)
        desktops_school   = st.slider("Desktops per School", 0.0, 50.0,
                                       float(panel["desktops_per_school"].mean()), 0.5)

    with col3:
        teach_per_school  = st.slider("Teachers per School", 3.0, 25.0,
                                       float(panel["teachers_per_school"].mean()), 0.5)
        nongovt_share     = st.slider("Non-Govt School Share", 0.0, 0.5,
                                       float(panel["nongovt_share"].mean()), 0.01)
        lag1              = st.slider("Last Year CSEE Pass Rate (%)", 60.0, 95.0,
                                       float(panel["csee_pass_rate"].dropna().mean()), 0.5)

    if st.button("Generate Prediction", type="primary"):
        # Build feature vector using mean for missing features
        mean_vals = panel[model_feats].mean()
        x_dict = dict(mean_vals)
        x_dict.update({
            "ptr_national":            ptr_nat,
            "qualified_teacher_ratio": qual_teach_ratio,
            "dropout_rate_pct":        dropout_rate,
            "gross_completion_rate":   completion_rate,
            "pct_schools_electricity": pct_electricity,
            "desktops_per_school":     desktops_school,
            "teachers_per_school":     teach_per_school,
            "nongovt_share":           nongovt_share,
            "csee_pass_rate_lag1":     lag1,
        })

        x_df  = pd.DataFrame([x_dict])[model_feats].fillna(mean_vals)
        x_sc  = scaler.transform(x_df.values)
        pred  = float(model.predict(x_sc)[0])
        pred  = min(100.0, max(0.0, pred))

        st.success(f"**Predicted CSEE Pass Rate: {pred:.2f}%**")

        # Context gauge
        national_mean = float(panel["csee_pass_rate"].dropna().mean())
        delta         = pred - national_mean
        col_a, col_b = st.columns(2)
        col_a.metric("Prediction", f"{pred:.2f}%",
                     delta=f"{delta:+.2f}pp vs national mean")
        col_b.metric("National Mean (2020-2024)", f"{national_mean:.2f}%")


# ===========================================================================
# PAGE 4: Forecast
# ===========================================================================
elif page == "Forecast (2025-2030)":
    st.title("📈 CSEE Pass Rate Forecast (2025-2030)")

    try:
        fcast_df = pd.read_csv(PATHS.table("csee_forecast_2025_2030.csv"))
        st.dataframe(fcast_df.round(2), use_container_width=True)
        st.download_button("Download Forecast Table (CSV)",
                           fcast_df.to_csv(index=False),
                           file_name="csee_forecast_2025_2030.csv")

        # Plot
        historical = csee_nat.set_index("year")["csee_pass_rate"].dropna()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical.index, historical.values, "o-", color="steelblue",
                lw=2.5, ms=8, label="Historical", zorder=5)
        colours = ["#e74c3c", "#2ecc71", "#f39c12"]
        for col, colour in zip(["Polynomial_Trend", "Optimistic", "Pessimistic"],
                               colours):
            if col in fcast_df.columns:
                ax.plot(fcast_df["Year"], fcast_df[col], "s--", color=colour,
                        lw=2, ms=7, label=col.replace("_", " "), alpha=0.85)
        ax.axvline(historical.index.max() + 0.5, color="gray", lw=1.5, linestyle=":")
        ax.set_xlabel("Year")
        ax.set_ylabel("CSEE Pass Rate (%)")
        ax.set_title("Tanzania CSEE Pass Rate Forecast (2025-2030)", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        render_fig(fig)

    except FileNotFoundError:
        st.info("Forecast not yet generated. Run Notebook 08 first.")


# ===========================================================================
# PAGE 5: Data Explorer
# ===========================================================================
elif page == "Data Explorer":
    st.title("🔍 Data Explorer")

    st.sidebar.subheader("Filters")
    year_filter   = st.sidebar.multiselect("Years",
                                           sorted(panel["year"].unique()),
                                           default=sorted(panel["year"].unique()))
    region_filter = st.sidebar.multiselect("Regions",
                                           sorted(panel["region"].unique()),
                                           default=[])

    df_filtered = panel[panel["year"].isin(year_filter)]
    if region_filter:
        df_filtered = df_filtered[df_filtered["region"].isin(region_filter)]

    st.markdown(f"**Showing {len(df_filtered)} rows** "
                f"({len(df_filtered['region'].unique())} regions, "
                f"{len(df_filtered['year'].unique())} years)")

    display_cols = st.multiselect(
        "Select columns to display",
        list(df_filtered.columns),
        default=[c for c in ["year", "region", "total_schools", "enrolment_f1f4",
                              "total_teachers", "ptr_regional", "pct_schools_electricity",
                              "csee_pass_rate"] if c in df_filtered.columns],
    )
    st.dataframe(df_filtered[display_cols].round(3), use_container_width=True)
    st.download_button("Download Filtered Data (CSV)",
                       df_filtered.to_csv(index=False),
                       file_name="best_filtered_data.csv")


# ===========================================================================
# PAGE 6: Model Information
# ===========================================================================
elif page == "Model Information":
    st.title("🎯 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")
        try:
            cv_df = pd.read_csv(PATHS.table("cv_model_comparison.csv"))
            st.dataframe(cv_df[["Model","CV_MAE_mean","CV_R2_mean","Fit_Time_s"]].round(4),
                         use_container_width=True)
        except FileNotFoundError:
            st.info("Run Notebook 05 to see model comparison.")

    with col2:
        st.subheader("Feature Importance")
        try:
            fi_df = pd.read_csv(PATHS.table("feature_importance_combined.csv"))
            top10 = fi_df.head(10)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(top10["feature"], top10["gb_importance"].fillna(0),
                    color="steelblue", alpha=0.85)
            ax.invert_yaxis()
            ax.set_xlabel("GB Feature Importance")
            ax.set_title("Top 10 Features", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            render_fig(fig)
        except FileNotFoundError:
            st.info("Run Notebook 07 to see feature importance.")

    st.subheader("Model Description")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | **Algorithm** | Gradient Boosting Regressor |
    | **Tuning** | GridSearchCV with Leave-One-Year-Out CV |
    | **Train period** | 2020-2023 (4 years x 26 regions) |
    | **Test period** | 2024 (temporal hold-out) |
    | **Target** | CSEE National Pass Rate (%) |
    | **Library** | scikit-learn |
    | **Author** | Habil Masawika |
    """)

    st.subheader("Feature List")
    if model_feats:
        st.write(", ".join(model_feats))
