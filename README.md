# 🎓 Forecasting Tanzania CSEE Pass Rates with Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

**An end-to-end machine learning pipeline for forecasting secondary education outcomes in Tanzania using five years of national administrative data (BEST 2020–2024).**

[Overview](#overview) · [Results](#results) · [Installation](#installation) · [Usage](#usage) · [Structure](#structure)

</div>

---

## Overview

Tanzania has invested heavily in secondary education expansion since 2015, with student enrolment roughly doubling and CSEE pass rates rising from **68% (2020) to 89% (2024)**. This project asks: *which measurable inputs drive those outcomes, and what does the trajectory look like through 2030?*

The pipeline covers the full data science lifecycle:

- **Data ingestion** from five Ministry of Education BEST annual reports (Excel workbooks with heterogeneous formats)
- **Cleaning and harmonisation** of 26-region × 5-year panel data
- **Exploratory analysis** with 20+ publication-quality visualisations
- **Feature engineering** producing 30+ lag, rolling, growth-rate, and composite features
- **Training and comparison** of 12 scikit-learn regression models with Leave-One-Year-Out cross-validation
- **Explainability** via permutation importance, partial dependence plots, and ICE curves
- **Forecasting** to 2030 under optimistic, baseline, and pessimistic scenarios
- **Interactive Streamlit dashboard** for prediction and exploration

---

## Results

### Model Performance (Cross-Validated MAE)

| Rank | Model | CV MAE | CV Std |
|------|-------|--------|--------|
| 🥇 | Linear Regression | **0.81** | ±0.38 |
| 🥈 | Ridge | 0.83 | ±0.32 |
| 🥉 | Lasso | 0.93 | ±1.12 |
| 4 | Random Forest | 1.10 | ±0.78 |
| 5 | Gradient Boosting | 1.19 | ±0.73 |

> Linear Regression and Ridge outperform tree ensembles — consistent with small-sample panel data where regularised linear models avoid overfitting.

### Key Predictors (converged across all XAI methods)

1. **Lagged CSEE pass rate** — strong path dependency / momentum
2. **Qualified teacher ratio** — teacher quality is the strongest modifiable input
3. **Gross completion rate** — student retention drives exam outcomes
4. **National PTR** — class size pressure matters at national scale
5. **Infrastructure quality index** — composite of electricity access + ICT penetration

### CSEE Pass Rate Forecast (2025–2030)

| Year | Pessimistic | Baseline | Optimistic |
|------|-------------|----------|------------|
| 2025 | 89.6% | **90.2%** | 90.9% |
| 2026 | 89.8% | **91.0%** | 92.4% |
| 2027 | 90.0% | **91.8%** | 93.9% |
| 2028 | 90.2% | **92.6%** | 95.4% |
| 2029 | 90.4% | **93.4%** | 96.9% |
| 2030 | 90.6% | **94.2%** | 98.4% |

All methods project continued improvement, converging on **92–94% by 2030** under the baseline scenario.

---

## Dataset

| Property | Detail |
|----------|--------|
| Source | Ministry of Education, Science and Technology (MoEST) — BEST Annual Reports |
| Years | 2020, 2021, 2022, 2023, 2024 |
| Coverage | 26 mainland Tanzania regions × 5 years = ~130 panel observations |
| Key outcome | CSEE national pass rate (%) |
| Key features | PTR, qualified teacher ratio, electricity access, ICT penetration, completion rate, dropout rate, school count, enrolment, teacher count |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/habilm-analytics/tanzania-best-ml-forecasting.git
cd tanzania-best-ml-forecasting
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# or
venv\Scripts\activate.bat       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place BEST data files

Copy the five BEST Excel workbooks into `data/raw/`:

```
data/raw/BEST_2020_National_Data.xlsx
data/raw/BEST_2021_National_Data.xlsx
data/raw/BEST_2022_National_Data.xlsx
data/raw/BEST_2023_National_Data.xlsx
data/raw/BEST_2024_National_Data.xlsx
```

---

## Usage

### Run notebooks in sequence

```bash
jupyter notebook
```

Execute notebooks `01` through `09` in order. Each saves outputs (processed data, models, figures) for use by subsequent notebooks.

| Notebook | Description | Runtime |
|----------|-------------|---------|
| 01 | Data Understanding: workbook inspection, sheet mapping | ~2 min |
| 02 | Data Cleaning: extraction, imputation, quality report | ~3 min |
| 03 | EDA: 20+ visualisations, PCA, K-means clustering | ~5 min |
| 04 | Feature Engineering: 30+ engineered features | ~2 min |
| 05 | Model Training: 12 models, Leave-One-Year-Out CV | ~8 min |
| 06 | Model Evaluation: residuals, learning curves | ~4 min |
| 07 | Explainable AI: PDP, ICE, permutation importance | ~5 min |
| 08 | Forecasting: 2025–2030 scenarios | ~3 min |
| 09 | Dashboard: launch guide and static preview | ~1 min |

### Launch the interactive dashboard

```bash
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` in your browser.

---

## Structure

```
tanzania-best-ml-forecasting/
│
├── data/
│   ├── raw/                    Raw BEST Excel workbooks (not committed)
│   ├── processed/              Cleaned and engineered datasets
│   │   ├── best_panel_cleaned.csv
│   │   ├── best_panel_features.csv
│   │   ├── csee_national_trend.csv
│   │   ├── completion_rate.csv
│   │   └── dropout_national.csv
│   └── external/               Placeholder for additional external data
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training.ipynb
│   ├── 06_Model_Evaluation.ipynb
│   ├── 07_Explainable_AI.ipynb
│   ├── 08_Forecasting.ipynb
│   └── 09_Dashboard.ipynb
│
├── src/
│   ├── data_loader.py          BESTLoader class — all extraction functions
│   ├── preprocessing.py        BESTCleaner class — imputation, quality report
│   ├── feature_engineering.py  FeatureEngineer class — all feature creation
│   ├── models.py               ModelEvaluator — model registry, tuning utilities
│   ├── evaluation.py           Residual plots, learning curves, comparison charts
│   ├── explainability.py       Feature importance, PDP, ICE, permutation importance
│   ├── forecasting.py          RecursiveForecaster — scenario modelling, forecast plots
│   ├── visualization.py        50+ publication-quality visualisation functions
│   └── utilities.py            Logging, paths, export helpers, reproducibility tools
│
├── models/                     Serialised trained models (.pkl)
│   ├── gradient_boosting.pkl
│   ├── gradient_boosting_tuned.pkl
│   ├── random_forest.pkl
│   ├── ridge.pkl
│   ├── lasso.pkl
│   ├── elasticnet.pkl
│   ├── extra_trees.pkl
│   ├── histgradientboosting.pkl
│   ├── adaboost.pkl
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   ├── feature_scaler.pkl
│   ├── baseline_mean.pkl
│   └── baseline_median.pkl
│
├── outputs/
│   ├── figures/                All publication-quality figures (150 dpi PNG)
│   ├── tables/                 CSV comparison and importance tables
│   ├── reports/                Plain-text analysis reports
│   └── dashboard/              Dashboard asset cache
│
├── app/
│   └── streamlit_app.py        Interactive Streamlit dashboard (6 pages)
│
├── requirements.txt
├── README.md
└── build_notebooks.py          Script used to generate the notebook suite
```

---

## Technologies

| Category | Tools |
|----------|-------|
| Data engineering | pandas, numpy, openpyxl, xlrd |
| Visualisation | matplotlib, seaborn |
| Machine learning | scikit-learn (12 models) |
| Model persistence | joblib |
| Statistics | scipy |
| Dashboard | Streamlit |
| Notebooks | Jupyter |

---

## Future Work

1. **Regional CSEE data** — MoEST publishing region-level pass rates would transform this into a true spatial panel regression
2. **Socioeconomic covariates** — NBS/TZNPS household survey data on income, distance to school, and parental education
3. **XGBoost / LightGBM / CatBoost** — stubbed in `models.py`; expected 5–10% MAE improvement
4. **SHAP integration** — full Shapley value decomposition for per-observation explanation
5. **Time-series models** — ARIMA / LSTM / Prophet as the data time series grows beyond 2024
6. **Causal inference** — DiD or synthetic control designs to isolate causal effects of specific policy interventions

---

## Author

**Habil Masawika**  
Senior Statistician, TARURA (Tanzania Rural and Urban Roads Agency)  
Founder, [Masawika AI Lab](https://github.com/habilm-analytics) — Data Science & AI for East Africa

- 🔗 LinkedIn: [linkedin.com/in/habil-masawika-177388403](https://www.linkedin.com/in/habil-masawika-177388403/)
- 📸 Instagram: [@habil_masawika](https://www.instagram.com/habil_masawika/)
- 💻 GitHub: [github.com/habilm-analytics](https://github.com/habilm-analytics)

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- **Ministry of Education, Science and Technology, Tanzania** for publishing annual BEST reports as open administrative data
- **NECTA** (National Examinations Council of Tanzania) for examination statistics
- The `scikit-learn`, `pandas`, `matplotlib`, and `seaborn` open-source communities

