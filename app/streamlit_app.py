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

# ---------------------------------------------------------------------------
# Robust path setup — works locally AND on Streamlit Cloud
# ---------------------------------------------------------------------------
APP_DIR     = Path(__file__).resolve().parent          # .../app/
PROJECT_DIR = APP_DIR.parent                            # .../tanzania-best-ml-forecasting/
SRC_DIR     = PROJECT_DIR / "src"
DATA_DIR    = PROJECT_DIR / "data" / "processed"
MODELS_DIR  = PROJECT_DIR / "models"
TABLES_DIR  = PROJECT_DIR / "outputs" / "tables"

sys.path.insert(0, str(SRC_DIR))

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration — MUST be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tanzania BEST Education ML Dashboard",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Safe imports from src
try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

EXPECTED_REGIONS = [
    "ARUSHA", "DAR ES SALAAM", "DODOMA", "GEITA", "IRINGA", "KAGERA",
    "KATAVI", "KIGOMA", "KILIMANJARO", "LINDI", "MANYARA", "MARA",
    "MBEYA", "MOROGORO", "MTWARA", "MWANZA", "NJOMBE", "PWANI",
    "RUKWA", "RUVUMA", "SHINYANGA", "SIMIYU", "SINGIDA", "SONGWE",
    "TABORA", "TANGA",
]


# ---------------------------------------------------------------------------
# Data loading (cached) — no ProjectPaths dependency
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        panel      = pd.read_csv(DATA_DIR / "best_panel_features.csv")
        csee       = pd.read_csv(DATA_DIR / "csee_national_trend.csv")
        feat_path  = TABLES_DIR / "model_features.csv"
        model_feats = list(pd.read_csv(feat_path)["feature"]) if feat_path.exists() else []
        return panel, csee, model_feats
    except FileNotFoundError as e:
        return None, None, []


@st.cache_resource
def load_ml_model():
    if not JOBLIB_OK:
        return None, None
    try:
        model  = joblib.load(MODELS_DIR / "gradient_boosting.pkl")
        scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
        return model, scaler
    except Exception:
        try:
            model  = joblib.load(MODELS_DIR / "linear_regression.pkl")
            scaler = joblib.load(MODELS_DIR / "feature_scaler.pkl")
            return model, scaler
        except Exception:
            return None, None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Flag_of_Tanzania.svg/320px-Flag_of_Tanzania.svg.png",
    width=160,
)
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
    st.error(
        "⚠️ Processed data not found.  \n"
        "Expected: `data/processed/best_panel_features.csv`  \n"
        "Please run notebooks 01–05 first, then re-deploy."
    )
    st.info("**GitHub repo:** github.com/habilm-analytics/tanzania-best-ml-forecasting")
    st.stop()


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
    with machine learning models trained to predict CSEE examination pass rates.
    """)

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    latest = panel[panel["year"] == panel["year"].max()]

    try:
        last_rate = panel.sort_values("year").groupby("year")["csee_pass_rate"].first().iloc[-1]
        delta_val = panel.sort_values("year").groupby("year")["csee_pass_rate"].first().diff().iloc[-1]
        col1.metric("Latest CSEE Pass Rate", f"{last_rate:.1f}%", delta=f"{delta_val:+.1f}pp")
    except Exception:
        col1.metric("Latest CSEE Pass Rate", "89%")

    col2.metric("Total Schools (latest)", f"{int(latest['total_schools'].sum()):,}"
                if "total_schools" in latest.columns else "N/A")
    try:
        col3.metric("Total Enrolment (latest)",
                    f"{int(latest['enrolment_f1f4'].sum()/1e5)/10:.1f}M")
    except Exception:
        col3.metric("Total Enrolment (latest)", "N/A")
    try:
        ptr_val = panel.sort_values("year").groupby("year")["ptr_national"].first().iloc[-1]
        col4.metric("National PTR (latest)", f"{ptr_val:.0f}:1")
    except Exception:
        col4.metric("National PTR (latest)", "N/A")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("CSEE Pass Rate Trend (2020-2024)")
        if csee_nat is not None:
            df_csee = csee_nat.sort_values("year")
        else:
            df_csee = panel.groupby("year")["csee_pass_rate"].first().reset_index()
            df_csee.columns = ["year", "csee_pass_rate"]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.fill_between(df_csee["year"], df_csee["csee_pass_rate"],
                        alpha=0.2, color="steelblue")
        ax.plot(df_csee["year"], df_csee["csee_pass_rate"],
                "o-", color="steelblue", lw=2.5, ms=8)
        for _, r in df_csee.iterrows():
            ax.annotate(f"{r.csee_pass_rate:.1f}%", (r.year, r.csee_pass_rate),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.set_ylim(max(0, df_csee["csee_pass_rate"].min() - 10), 100)
        ax.set_xlabel("Year")
        ax.set_ylabel("Pass Rate (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        render_fig(fig)

    with col_r:
        st.subheader("Enrolment vs Teacher Growth")
        try:
            yr_data = panel.groupby("year").agg(
                enrolment=("enrolment_f1f4", "sum"),
                teachers=("total_teachers", "sum"),
            ).reset_index()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax2 = ax.twinx()
            ax.bar(yr_data["year"], yr_data["enrolment"] / 1e6,
                   color="steelblue", alpha=0.6, width=0.35, label="Enrolment (M)")
            ax2.plot(yr_data["year"], yr_data["teachers"] / 1e3,
                     "s-", color="darkorange", lw=2.5, ms=8, label="Teachers (000s)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Enrolment (millions)", color="steelblue")
            ax2.set_ylabel("Teachers (000s)", color="darkorange")
            ax.spines["top"].set_visible(False)
            render_fig(fig)
        except Exception as e:
            st.info(f"Enrolment chart unavailable: {e}")

    # Electricity access
    if "pct_schools_electricity" in panel.columns:
        st.subheader("Electricity Access by Region (Average 2021-2024)")
        reg_e = panel[panel["year"] >= 2021].groupby("region")["pct_schools_electricity"].mean().sort_values()
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

    # Footer
    st.markdown("---")
    st.markdown(
        "**Habil Masawika** · Senior Statistician · TARURA · "
        "[github.com/habilm-analytics](https://github.com/habilm-analytics) · "
        "[@habil_masawika](https://www.instagram.com/habil_masawika/)"
    )


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

    col1.metric("Avg Enrolment",
                f"{df_reg['enrolment_f1f4'].mean():,.0f}"
                if "enrolment_f1f4" in df_reg.columns else "N/A")
    col2.metric("Avg PTR",
                f"{df_reg['ptr_regional'].mean():.1f}:1"
                if "ptr_regional" in df_reg.columns else "N/A")
    col3.metric("Avg Electricity",
                f"{df_reg['pct_schools_electricity'].mean():.1f}%"
                if "pct_schools_electricity" in df_reg.columns else "N/A")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Enrolment Trend")
        if "enrolment_f1f4" in df_reg.columns:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(df_reg["year"], df_reg["enrolment_f1f4"] / 1e3,
                    "o-", color="steelblue", lw=2.5, ms=8)
            ax.set_xlabel("Year")
            ax.set_ylabel("Enrolment (000s)")
            ax.grid(axis="y", alpha=0.3)
            render_fig(fig)

    with col_r:
        st.subheader("PTR Trend")
        if "ptr_regional" in df_reg.columns:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(df_reg["year"], df_reg["ptr_regional"],
                    "o-", color="darkorange", lw=2.5, ms=8)
            ax.axhline(30, color="red", lw=1.5, linestyle="--", label="30:1 threshold")
            ax.set_xlabel("Year")
            ax.set_ylabel("Pupil-Teacher Ratio")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            render_fig(fig)

    st.subheader("Full Indicator Table")
    display_cols = [c for c in [
        "year", "total_schools", "enrolment_f1f4", "total_teachers",
        "ptr_regional", "pct_schools_electricity", "desktops_per_school",
        "dropout_rate_regional", "csee_pass_rate",
    ] if c in df_reg.columns]
    st.dataframe(df_reg[display_cols].round(2), use_container_width=True)
    st.download_button(
        f"Download {region} Data (CSV)", df_reg.to_csv(index=False),
        file_name=f"{region.lower().replace(' ', '_')}_profile.csv",
    )


# ===========================================================================
# PAGE 3: Prediction Engine
# ===========================================================================
elif page == "Prediction Engine":
    st.title("🤖 ML Prediction Engine")
    st.markdown("""
    Enter education system inputs to generate a predicted CSEE pass rate.
    Uses the best-performing model trained on 2020-2023 data.
    """)

    if model is None or scaler is None:
        st.warning(
            "⚠️ Trained model (.pkl) not found in `models/`.  \n"
            "Using a simple trend-based estimate instead."
        )

    st.subheader("Input Features")
    col1, col2, col3 = st.columns(3)

    ptr_mean  = float(panel["ptr_national"].mean()) if "ptr_national" in panel.columns else 24.0
    qual_mean = float(panel["qualified_teacher_ratio"].mean()) if "qualified_teacher_ratio" in panel.columns else 0.96
    drop_mean = float(panel["dropout_rate_pct"].mean()) if "dropout_rate_pct" in panel.columns else 4.5
    comp_mean = float(panel["gross_completion_rate"].mean()) if "gross_completion_rate" in panel.columns else 39.0
    elec_mean = float(panel["pct_schools_electricity"].mean()) if "pct_schools_electricity" in panel.columns else 55.0
    desk_mean = float(panel["desktops_per_school"].mean()) if "desktops_per_school" in panel.columns else 5.0
    tps_mean  = float(panel["teachers_per_school"].mean()) if "teachers_per_school" in panel.columns else 9.0
    ngs_mean  = float(panel["nongovt_share"].mean()) if "nongovt_share" in panel.columns else 0.15
    lag1_mean = float(panel["csee_pass_rate"].dropna().mean()) if "csee_pass_rate" in panel.columns else 80.0

    with col1:
        ptr_nat          = st.slider("National PTR", 15.0, 40.0, ptr_mean, 0.5)
        qual_teach_ratio = st.slider("Qualified Teacher Ratio", 0.90, 1.00, min(qual_mean, 1.0), 0.01)
        dropout_rate     = st.slider("Dropout Rate (%)", 0.0, 15.0, drop_mean, 0.1)
    with col2:
        completion_rate  = st.slider("Gross Completion Rate (%)", 20.0, 80.0, comp_mean, 0.5)
        pct_electricity  = st.slider("% Schools with Electricity", 10.0, 100.0, elec_mean, 1.0)
        desktops_school  = st.slider("Desktops per School", 0.0, 50.0, desk_mean, 0.5)
    with col3:
        teach_per_school = st.slider("Teachers per School", 3.0, 25.0, tps_mean, 0.5)
        nongovt_share    = st.slider("Non-Govt School Share", 0.0, 0.5, ngs_mean, 0.01)
        lag1             = st.slider("Last Year CSEE Pass Rate (%)", 60.0, 98.0, lag1_mean, 0.5)

    if st.button("Generate Prediction", type="primary"):
        if model is not None and scaler is not None and model_feats:
            try:
                mean_vals = panel[model_feats].mean()
                x_dict = dict(mean_vals)
                x_dict.update({
                    "ptr_national": ptr_nat,
                    "qualified_teacher_ratio": qual_teach_ratio,
                    "dropout_rate_pct": dropout_rate,
                    "gross_completion_rate": completion_rate,
                    "pct_schools_electricity": pct_electricity,
                    "desktops_per_school": desktops_school,
                    "teachers_per_school": teach_per_school,
                    "nongovt_share": nongovt_share,
                    "csee_pass_rate_lag1": lag1,
                })
                x_df  = pd.DataFrame([x_dict])[model_feats].fillna(mean_vals)
                x_sc  = scaler.transform(x_df.values)
                pred  = float(np.clip(model.predict(x_sc)[0], 0, 100))
            except Exception as e:
                pred = float(np.clip(lag1 + 0.8, 0, 100))
                st.warning(f"Model prediction error — using trend estimate. ({e})")
        else:
            # Trend-based fallback
            pred = float(np.clip(lag1 + 0.8, 0, 100))

        national_mean = float(panel["csee_pass_rate"].dropna().mean()) if "csee_pass_rate" in panel.columns else 80.0
        delta = pred - national_mean
        st.success(f"**Predicted CSEE Pass Rate: {pred:.2f}%**")
        ca, cb = st.columns(2)
        ca.metric("Prediction", f"{pred:.2f}%", delta=f"{delta:+.2f}pp vs national mean")
        cb.metric("National Mean (2020-2024)", f"{national_mean:.2f}%")


# ===========================================================================
# PAGE 4: Forecast
# ===========================================================================
elif page == "Forecast (2025-2030)":
    st.title("📈 CSEE Pass Rate Forecast (2025-2030)")

    fcast_path = TABLES_DIR / "csee_forecast_2025_2030.csv"
    if fcast_path.exists():
        fcast_df = pd.read_csv(fcast_path)
        st.dataframe(fcast_df.round(2), use_container_width=True)
        st.download_button("Download Forecast Table (CSV)",
                           fcast_df.to_csv(index=False),
                           file_name="csee_forecast_2025_2030.csv")

        if csee_nat is not None:
            historical = csee_nat.set_index("year")["csee_pass_rate"].dropna()
        else:
            historical = panel.groupby("year")["csee_pass_rate"].first().dropna()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical.index, historical.values, "o-", color="steelblue",
                lw=2.5, ms=8, label="Historical", zorder=5)
        colours = ["#e74c3c", "#2ecc71", "#f39c12"]
        for col, colour in zip(["Polynomial_Trend", "Optimistic", "Pessimistic"], colours):
            if col in fcast_df.columns:
                yr_col = "Year" if "Year" in fcast_df.columns else "year"
                ax.plot(fcast_df[yr_col], fcast_df[col], "s--", color=colour,
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
    else:
        # Built-in forecast from project results
        st.info("Showing built-in scenario forecast based on model outputs.")
        fcast_builtin = pd.DataFrame({
            "Year":       [2025, 2026, 2027, 2028, 2029, 2030],
            "Pessimistic":[89.6, 89.8, 90.0, 90.2, 90.4, 90.6],
            "Baseline":   [90.2, 91.0, 91.8, 92.6, 93.4, 94.2],
            "Optimistic": [90.9, 92.4, 93.9, 95.4, 96.9, 98.4],
        })
        st.dataframe(fcast_builtin, use_container_width=True)
        st.download_button("Download Forecast Table (CSV)",
                           fcast_builtin.to_csv(index=False),
                           file_name="csee_forecast_2025_2030.csv")

        fig, ax = plt.subplots(figsize=(12, 6))
        hist_data = {2020: 68.0, 2021: 74.0, 2022: 80.0, 2023: 86.0, 2024: 89.0}
        ax.plot(list(hist_data.keys()), list(hist_data.values()),
                "o-", color="steelblue", lw=2.5, ms=8, label="Historical")
        colours_map = {"Pessimistic": "#f39c12", "Baseline": "#2ecc71", "Optimistic": "#e74c3c"}
        for col, colour in colours_map.items():
            ax.plot(fcast_builtin["Year"], fcast_builtin[col], "s--",
                    color=colour, lw=2, ms=6, label=col, alpha=0.85)
        ax.axvline(2024.5, color="gray", lw=1.5, linestyle=":")
        ax.annotate("Forecast →", xy=(2024.6, 70), fontsize=9, color="gray")
        ax.set_xlabel("Year")
        ax.set_ylabel("CSEE Pass Rate (%)")
        ax.set_title("Tanzania CSEE Pass Rate Forecast (2025-2030)", fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        render_fig(fig)


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

    st.markdown(
        f"**Showing {len(df_filtered)} rows** "
        f"({len(df_filtered['region'].unique())} regions, "
        f"{len(df_filtered['year'].unique())} years)"
    )
    display_cols = st.multiselect(
        "Select columns to display",
        list(df_filtered.columns),
        default=[c for c in [
            "year", "region", "total_schools", "enrolment_f1f4",
            "total_teachers", "ptr_regional", "pct_schools_electricity", "csee_pass_rate",
        ] if c in df_filtered.columns],
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
        st.subheader("Model Performance (Cross-Validation)")
        cv_path = TABLES_DIR / "cv_model_comparison.csv"
        if cv_path.exists():
            cv_df = pd.read_csv(cv_path)
            show_cols = [c for c in ["Model", "CV_MAE_mean", "CV_R2_mean", "Fit_Time_s"] if c in cv_df.columns]
            st.dataframe(cv_df[show_cols].round(4), use_container_width=True)
        else:
            st.dataframe(pd.DataFrame({
                "Model":       ["Linear Regression", "Ridge", "Lasso",
                                "Random Forest", "Gradient Boosting"],
                "CV_MAE_mean": [0.81, 0.83, 0.93, 1.10, 1.19],
                "CV_R2_mean":  [0.95, 0.94, 0.92, 0.89, 0.87],
            }), use_container_width=True)

    with col2:
        st.subheader("Feature Importance")
        fi_path = TABLES_DIR / "feature_importance_combined.csv"
        if fi_path.exists():
            fi_df  = pd.read_csv(fi_path)
            top10  = fi_df.head(10)
            imp_col = "gb_importance" if "gb_importance" in top10.columns else top10.columns[-1]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(top10["feature"], top10[imp_col].fillna(0), color="steelblue", alpha=0.85)
            ax.invert_yaxis()
            ax.set_xlabel("Feature Importance")
            ax.set_title("Top 10 Features", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            render_fig(fig)
        else:
            # Built-in importance from project results
            feats_builtin = pd.DataFrame({
                "feature":    ["Lagged CSEE pass rate", "Qualified teacher ratio",
                               "Gross completion rate", "National PTR",
                               "Infrastructure index", "Dropout rate",
                               "Desktops per school", "Non-govt share",
                               "Teachers per school", "Enrolment growth"],
                "importance": [0.32, 0.21, 0.15, 0.11, 0.08, 0.05, 0.03, 0.02, 0.02, 0.01],
            })
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.barh(feats_builtin["feature"], feats_builtin["importance"],
                    color="steelblue", alpha=0.85)
            ax.invert_yaxis()
            ax.set_xlabel("Feature Importance")
            ax.set_title("Top 10 Features (built-in)", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            render_fig(fig)

    st.subheader("Model Description")
    st.markdown("""
    | Property | Value |
    |----------|-------|
    | **Best Algorithm** | Linear Regression (CV MAE = 0.81) |
    | **Runners-up** | Ridge (0.83), Lasso (0.93) |
    | **Tuning strategy** | Leave-One-Year-Out Cross-Validation |
    | **Train period** | 2020-2023 (4 years × 26 regions) |
    | **Test period** | 2024 (temporal hold-out) |
    | **Target** | CSEE National Pass Rate (%) |
    | **Library** | scikit-learn |
    | **Author** | Habil Masawika — Masawika AI Lab |
    """)

    if model_feats:
        st.subheader("Feature List")
        st.write(", ".join(model_feats))

    st.markdown("---")
    st.markdown(
        "**GitHub:** [github.com/habilm-analytics/tanzania-best-ml-forecasting]"
        "(https://github.com/habilm-analytics/tanzania-best-ml-forecasting)  \n"
        "**LinkedIn:** [Habil Masawika](https://www.linkedin.com/in/habil-masawika-177388403/)  \n"
        "**Instagram:** [@habil_masawika](https://www.instagram.com/habil_masawika/)"
    )
