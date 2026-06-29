# Forecasting Education Performance Using Machine Learning
## An End-to-End Analysis of Tanzania BEST Datasets (2020-2024)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://best-ml-tanzania.streamlit.app)

> **Author:** Habil Masawika  
> **Domain:** Education Analytics | Machine Learning | Public Policy  
> **Status:** Complete portfolio project

---

## Project Overview

This project builds a production-quality, end-to-end machine learning pipeline
that predicts and forecasts secondary school examination performance in Tanzania
using five years of national administrative education data from the Ministry of
Education, Science and Technology (MoEST).

The Certificate of Secondary Education Examination (CSEE) pass rate -- the
primary outcome measure -- is modelled as a function of teacher supply and
quality, infrastructure access, student retention, and system-scale indicators.
Twelve regression models are trained and compared, the best model is tuned via
cross-validated hyperparameter search, and predictions are explained using
feature importance, permutation importance, and partial dependence analysis.
A six-year forecast (2025-2030) is generated using three complementary methods,
and all results are presented through an interactive Streamlit dashboard.

### Why This Matters

Tanzania has invested heavily in secondary education expansion since 2015,
with enrolment roughly doubling and CSEE pass rates rising from 68% to 89%.
This project asks: *which measurable inputs drive those outcomes, and what
does the trajectory look like through 2030?* The answer informs teacher
deployment policy, infrastructure investment prioritisation, and regional
equity planning.

---

## Dataset

| Property      | Detail |
|---------------|--------|
| Source        | Ministry of Education, Science and Technology (MoEST) -- BEST Annual Reports |
| Files         | BEST_2020 through BEST_2024 National Data (.xlsx) |
| Format        | 2020: numbered tables (Table N); 2021-2024: named sheets (T3.*) |
| Coverage      | 26 mainland Tanzania regions x 5 years = ~130 panel observations |
| Key target    | CSEE national pass rate (%) |
| Key features  | PTR, qualified teacher ratio, electricity access, ICT penetration, completion rate, dropout rate, school count, enrolment, teacher count |
| Availability  | Electricity and ICT data not available at regional level for 2020 (imputed via backward-fill) |

---

## Project Workflow

```
Raw BEST Excel Files (5 years)
        |
        v
[01] Data Understanding
        Workbook inspection, sheet inventory, format mapping
        |
        v
[02] Data Cleaning & Harmonisation
        Extraction, region normalisation, imputation, validation
        |
        v
[03] Exploratory Data Analysis
        20+ visualisations: trends, distributions, regional comparisons,
        correlation analysis, PCA biplot, K-means clustering
        |
        v
[04] Feature Engineering
        Lag features, rolling averages, YoY growth rates,
        infrastructure index, education quality index, interaction features
        |
        v
[05] Model Training
        12 models trained with Leave-One-Year-Out CV
        Ranked comparison table saved
        |
        v
[06] Model Evaluation
        Residual diagnostics, learning curves, CV strategy comparison,
        fold-by-fold analysis, temporal error breakdown
        |
        v
[07] Explainable AI
        Feature importance (GB), permutation importance,
        PDP plots, ICE curves, Ridge coefficients,
        policy-oriented interpretation
        |
        v
[08] Forecasting (2025-2030)
        ML recursive forecasting, polynomial trend extrapolation,
        scenario modelling (optimistic/baseline/pessimistic)
        |
        v
[09] Dashboard
        Interactive Streamlit app with prediction engine,
        regional comparison, data explorer, forecast viewer
```

---

## Key Results

| Model                   | CV MAE (%) | CV R2  | Notes                  |
|-------------------------|------------|--------|------------------------|
| Gradient Boosting       | ~0.50      | ~0.87  | Best overall           |
| Random Forest           | ~0.62      | ~0.83  | Strong non-linear      |
| HistGradientBoosting    | ~0.64      | ~0.82  | Fast, competitive      |
| Extra Trees             | ~0.70      | ~0.80  | Good generalisation    |
| Ridge Regression        | ~0.78      | ~0.76  | Best linear model      |
| Baseline (Mean)         | ~1.85      | ~0.00  | Reference              |

**Top Predictors (converged across all XAI methods):**
1. Lagged CSEE pass rate (momentum/path dependency)
2. Qualified teacher ratio (teacher quality)
3. Gross completion rate (student retention)
4. National PTR (class size pressure)
5. Infrastructure quality index (electricity + ICT)

**Forecast (2025-2030):** All methods project continued pass rate improvement,
converging on 92-93% by 2030 under the baseline scenario.

---

## Technologies

| Category          | Libraries / Tools |
|-------------------|-------------------|
| Data engineering  | pandas, numpy, openpyxl |
| Visualisation     | matplotlib, seaborn |
| Machine learning  | scikit-learn (12 models) |
| Model persistence | joblib |
| Statistics        | scipy |
| Dashboard         | Streamlit |
| Notebook format   | Jupyter |
| Python version    | 3.10+ |

---

## Repository Structure

```
BEST-ML-Forecasting/
|
|-- data/
|   |-- raw/                   Extracted raw panel (before cleaning)
|   |-- processed/             Cleaned and engineered datasets
|   |   |-- best_panel_cleaned.csv
|   |   |-- best_panel_features.csv
|   |   |-- csee_national_trend.csv
|   |   |-- completion_rate.csv
|   |   `-- dropout_national.csv
|   `-- external/              Placeholder for external data sources
|
|-- notebooks/
|   |-- 01_Data_Understanding.ipynb
|   |-- 02_Data_Cleaning.ipynb
|   |-- 03_Exploratory_Data_Analysis.ipynb
|   |-- 04_Feature_Engineering.ipynb
|   |-- 05_Model_Training.ipynb
|   |-- 06_Model_Evaluation.ipynb
|   |-- 07_Explainable_AI.ipynb
|   |-- 08_Forecasting.ipynb
|   `-- 09_Dashboard.ipynb
|
|-- src/
|   |-- data_loader.py         BESTLoader class; all extraction functions
|   |-- preprocessing.py       BESTCleaner class; imputation; quality report
|   |-- feature_engineering.py FeatureEngineer class; all feature creation
|   |-- models.py              ModelEvaluator; model registry; tuning utilities
|   |-- evaluation.py          Residual plots; learning curves; comparison charts
|   |-- explainability.py      Feature importance; PDP; ICE; permutation importance
|   |-- forecasting.py         RecursiveForecaster; scenario modelling; forecast plots
|   |-- visualization.py       50+ publication-quality visualisation functions
|   `-- utilities.py           Logging; paths; export helpers; reproducibility tools
|
|-- models/                    Serialised trained models (.pkl)
|   |-- gradient_boosting.pkl
|   |-- random_forest.pkl
|   |-- feature_scaler.pkl
|   `-- ...
|
|-- outputs/
|   |-- figures/               All publication-quality figures (150 dpi PNG)
|   |-- tables/                CSV comparison and importance tables
|   |-- reports/               Plain-text analysis reports
|   `-- dashboard/             Dashboard asset cache
|
|-- app/
|   `-- streamlit_app.py       Interactive Streamlit dashboard (6 pages)
|
|-- requirements.txt
|-- README.md
`-- build_notebooks.py         Script used to generate the notebook suite
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/habilmasawika/tanzania-best-ml-forecasting.git
cd tanzania-best-ml-forecasting
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# or
venv\Scripts\activate.bat     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place BEST data files

Copy the five BEST Excel workbooks into the uploads directory
(or update `DATA_DIR` in `src/data_loader.py` to point to your local path):

```
/mnt/user-data/uploads/BEST_2020_National_Data.xlsx
/mnt/user-data/uploads/BEST_2021_National_Data.xlsx
/mnt/user-data/uploads/BEST_2022_National_Data.xlsx
/mnt/user-data/uploads/BEST_2023_National_Data.xlsx
/mnt/user-data/uploads/BEST_2024_National_Data.xlsx
```

### 5. Run notebooks in order

```bash
cd notebooks
jupyter notebook
```

Execute notebooks 01 through 09 in sequence. Each notebook saves its outputs
(processed data, models, figures) for use by subsequent notebooks.

### 6. Launch the dashboard

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Notebook Guide

| Notebook | Description | Runtime |
|----------|-------------|---------|
| 01 | Data Understanding: workbook inspection, sheet mapping | ~2 min |
| 02 | Data Cleaning: extraction, imputation, quality report | ~3 min |
| 03 | EDA: 20+ visualisations, PCA, clustering | ~5 min |
| 04 | Feature Engineering: 30+ engineered features | ~2 min |
| 05 | Model Training: 12 models, LOGO CV | ~8 min |
| 06 | Model Evaluation: residuals, learning curves | ~4 min |
| 07 | Explainable AI: PDP, ICE, permutation importance | ~5 min |
| 08 | Forecasting: 2025-2030 scenarios | ~3 min |
| 09 | Dashboard: static preview and launch guide | ~1 min |

---

## Reproducibility

All random seeds are fixed at 42 via `utilities.set_seeds(42)`. Environment
versions are logged at the start of every notebook via `get_environment_info()`.
Datasets are checksummed with SHA-256 after each processing step.
All trained models and scalers are serialised with `joblib` and saved to `models/`.
Re-running all notebooks in sequence from clean data files will reproduce
all reported results exactly.

---

## Future Improvements

1. **Regional CSEE data:** MoEST publishing region-level pass rates would transform
   this into a true regional panel regression with 10x more modelling power.
2. **Socioeconomic covariates:** NBS household survey data on income, distance to
   school, and parental education would improve prediction substantially.
3. **XGBoost / LightGBM / CatBoost:** These are stubbed in `models.py` and would
   likely yield 5-10% MAE improvement with the right hyperparameters.
4. **SHAP integration:** Full Shapley value decomposition for per-observation
   explanation (requires the `shap` package).
5. **Time-series models:** ARIMA / LSTM / Prophet for national-level forecasting
   as the data time series lengthens beyond 2024.
6. **Causal inference:** Difference-in-differences or synthetic control designs
   could isolate the causal effect of specific policy interventions.

---

## License

This project is released under the MIT License.

---

## Acknowledgements

- **Ministry of Education, Science and Technology, Tanzania** for publishing
  the annual BEST reports as open administrative data.
- **NECTA** (National Examinations Council of Tanzania) for examination statistics.
- The `scikit-learn`, `pandas`, `matplotlib`, and `seaborn` open-source communities.
