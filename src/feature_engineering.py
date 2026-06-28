"""
feature_engineering.py
======================
Advanced feature engineering for Tanzania BEST education panel data.
Creates lag variables, rolling averages, growth rates, composite indices,
interaction features, and polynomial features.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

logger = logging.getLogger("feature_engineering")


class FeatureEngineer:
    """
    Generates advanced features from the cleaned BEST panel.

    All methods are chainable (return self) except build() which returns the
    engineered DataFrame.

    Parameters
    ----------
    panel : pd.DataFrame
        Cleaned panel from BESTCleaner.run().
    """

    # Target and core predictors
    TARGET = "csee_pass_rate"
    BASE_FEATURES = [
        "ptr_national", "qualified_teacher_ratio", "dropout_rate_pct",
        "gross_completion_rate", "pct_schools_electricity",
        "desktops_per_school", "teachers_per_school", "nongovt_share",
        "enrolment_per_school",
    ]

    def __init__(self, panel: pd.DataFrame):
        self.df = panel.copy().sort_values(["region", "year"]).reset_index(drop=True)
        self.feature_log: List[Dict] = []

    def _log(self, name: str, description: str) -> None:
        self.feature_log.append({"feature": name, "description": description})
        logger.debug(f"Feature created: {name}")

    # ------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------
    def add_lag_features(self, cols: Optional[List[str]] = None,
                         lags: List[int] = [1, 2]) -> "FeatureEngineer":
        """Create lag-n versions of key indicators."""
        if cols is None:
            cols = [
                "csee_pass_rate", "ptr_national", "gross_completion_rate",
                "pct_schools_electricity", "dropout_rate_pct",
                "enrolment_f1f4", "total_teachers",
            ]
        for col in cols:
            if col not in self.df.columns:
                continue
            for lag in lags:
                new_name = f"{col}_lag{lag}"
                self.df[new_name] = (
                    self.df.groupby("region")[col].shift(lag)
                )
                self._log(new_name,
                          f"Lag-{lag} of {col} by region (reflects prior-year state)")
        return self

    # ------------------------------------------------------------------
    # Rolling averages
    # ------------------------------------------------------------------
    def add_rolling_features(self, cols: Optional[List[str]] = None,
                             windows: List[int] = [2, 3]) -> "FeatureEngineer":
        """Create rolling window averages."""
        if cols is None:
            cols = ["csee_pass_rate", "ptr_national", "dropout_rate_pct",
                    "gross_completion_rate", "pct_schools_electricity"]
        for col in cols:
            if col not in self.df.columns:
                continue
            for w in windows:
                if len(self.df["year"].unique()) < w:
                    continue
                new_name = f"{col}_roll{w}"
                self.df[new_name] = (
                    self.df.groupby("region")[col]
                    .transform(lambda s: s.rolling(w, min_periods=1).mean())
                )
                self._log(new_name, f"Rolling {w}-year mean of {col}")
        return self

    # ------------------------------------------------------------------
    # Year-over-year growth rates
    # ------------------------------------------------------------------
    def add_growth_features(self) -> "FeatureEngineer":
        """Create year-over-year percentage change features."""
        growth_pairs = [
            ("enrolment_f1f4",           "enrolment_yoy_growth"),
            ("total_teachers",           "teacher_yoy_growth"),
            ("total_schools",            "school_yoy_growth"),
            ("pct_schools_electricity",  "electricity_yoy_growth"),
            ("desktop_computers",        "ict_yoy_growth"),
        ]
        for col, new_name in growth_pairs:
            if col not in self.df.columns:
                continue
            self.df[new_name] = (
                self.df.groupby("region")[col].pct_change().round(4)
            )
            self._log(new_name, f"YoY % change in {col} by region")

        # National enrolment-teacher imbalance trend
        if "enrolment_yoy_growth" in self.df.columns and "teacher_yoy_growth" in self.df.columns:
            self.df["enrolment_teacher_growth_gap"] = (
                self.df["enrolment_yoy_growth"] - self.df["teacher_yoy_growth"]
            )
            self._log("enrolment_teacher_growth_gap",
                      "Difference between enrolment and teacher growth rates — positive = widening PTR pressure")
        return self

    # ------------------------------------------------------------------
    # Infrastructure composite index
    # ------------------------------------------------------------------
    def add_infrastructure_index(self) -> "FeatureEngineer":
        """
        Create a composite infrastructure quality index (0-1 scale).
        Combines electricity access and ICT penetration.
        """
        cols_to_use = []
        if "pct_schools_electricity" in self.df.columns:
            cols_to_use.append("pct_schools_electricity")
        if "desktops_per_school" in self.df.columns:
            cols_to_use.append("desktops_per_school")
        if "laptops_per_school" in self.df.columns:
            cols_to_use.append("laptops_per_school")

        if not cols_to_use:
            return self

        mm = MinMaxScaler()
        raw = self.df[cols_to_use].fillna(0)
        self.df["infra_index"] = mm.fit_transform(raw).mean(axis=1).round(4)
        self._log("infra_index",
                  "Composite infrastructure quality index (0=worst, 1=best) — average of normalised electricity and ICT indicators")

        # Electricity-only index
        if "pct_schools_electricity" in self.df.columns:
            self.df["electricity_index"] = (
                self.df["pct_schools_electricity"] / 100.0
            ).round(4)
            self._log("electricity_index", "Electricity access rate (0-1 scale)")

        # ICT-only index
        if "desktops_per_school" in self.df.columns:
            max_ict = self.df["desktops_per_school"].quantile(0.95)
            self.df["ict_index"] = (
                self.df["desktops_per_school"].clip(0, max_ict) / max_ict
            ).fillna(0).round(4)
            self._log("ict_index", "ICT penetration index (desktops per school, 0-1 scale)")

        return self

    # ------------------------------------------------------------------
    # Gender and equity features
    # ------------------------------------------------------------------
    def add_gender_features(self) -> "FeatureEngineer":
        """Create gender equity indicators where data is available."""
        # Girls' dropout proxied from total dropout and pregnancy rates
        if "total_dropouts" in self.df.columns and "enrolment_f1f4" in self.df.columns:
            self.df["dropout_burden"] = (
                self.df["total_dropouts"] / self.df["enrolment_f1f4"] * 100
            ).replace([np.inf, -np.inf], np.nan).round(4)
            self._log("dropout_burden",
                      "Regional dropout burden (dropouts / enrolment %) — proxy for retention challenge")

        if "disability_enrolment" in self.df.columns and "enrolment_f1f4" in self.df.columns:
            self.df["disability_inclusion_rate"] = (
                self.df["disability_enrolment"] / self.df["enrolment_f1f4"] * 1000
            ).replace([np.inf, -np.inf], np.nan).round(4)
            self._log("disability_inclusion_rate",
                      "Disabled students per 1,000 enrolled — inclusion proxy")
        return self

    # ------------------------------------------------------------------
    # School density and system scale
    # ------------------------------------------------------------------
    def add_system_scale_features(self) -> "FeatureEngineer":
        """Create system scale and density features."""
        if "nongovt_share" in self.df.columns:
            # Already created in preprocessing; ensure it exists
            pass

        # Year as a numeric feature for trend capture
        self.df["year_index"] = self.df["year"] - self.df["year"].min()
        self._log("year_index", "Years since start of panel (0=2020) — captures temporal trend")

        # Log-scaled enrolment (reduces right skew)
        if "enrolment_f1f4" in self.df.columns:
            self.df["log_enrolment"] = np.log1p(self.df["enrolment_f1f4"]).round(4)
            self._log("log_enrolment", "Log(1+enrolment) — reduces right skew for regression")

        if "total_schools" in self.df.columns:
            self.df["log_schools"] = np.log1p(self.df["total_schools"]).round(4)
            self._log("log_schools", "Log(1+total_schools)")

        return self

    # ------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------
    def add_interaction_features(self) -> "FeatureEngineer":
        """Create theoretically motivated interaction features."""
        # Teacher quality x availability
        if ("qualified_teacher_ratio" in self.df.columns and
                "teachers_per_school" in self.df.columns):
            self.df["teacher_quality_x_density"] = (
                self.df["qualified_teacher_ratio"] * self.df["teachers_per_school"]
            ).round(4)
            self._log("teacher_quality_x_density",
                      "Interaction: qualified teacher ratio x teachers per school")

        # Infrastructure x teacher quality
        if "infra_index" in self.df.columns and "qualified_teacher_ratio" in self.df.columns:
            self.df["infra_x_teacher_quality"] = (
                self.df["infra_index"] * self.df["qualified_teacher_ratio"]
            ).round(4)
            self._log("infra_x_teacher_quality",
                      "Interaction: infrastructure index x qualified teacher ratio")

        # PTR x electricity (overloaded + under-resourced schools)
        if "ptr_national" in self.df.columns and "electricity_index" in self.df.columns:
            self.df["ptr_x_electricity"] = (
                self.df["ptr_national"] * self.df["electricity_index"]
            ).round(4)
            self._log("ptr_x_electricity",
                      "Interaction: PTR x electricity access (high PTR + low electricity = double disadvantage)")

        # Completion x dropout (coherence check)
        if "gross_completion_rate" in self.df.columns and "dropout_rate_pct" in self.df.columns:
            self.df["retention_pressure"] = (
                self.df["dropout_rate_pct"] / (self.df["gross_completion_rate"] + 1e-9)
            ).round(4)
            self._log("retention_pressure",
                      "Ratio of dropout rate to completion rate — high = more students leaving than completing")

        return self

    # ------------------------------------------------------------------
    # Education quality index
    # ------------------------------------------------------------------
    def add_education_quality_index(self) -> "FeatureEngineer":
        """
        Composite education quality index combining teacher quality,
        infrastructure, and retention metrics.
        """
        components = []
        mm = MinMaxScaler()

        comp_cols = {
            "qualified_teacher_ratio": True,  # higher is better
            "gross_completion_rate":   True,
            "pct_schools_electricity": True,
            "ptr_national":            False,  # lower is better (inverted)
            "dropout_rate_pct":        False,  # lower is better
        }
        available = {k: v for k, v in comp_cols.items() if k in self.df.columns}
        if len(available) < 3:
            return self

        df_comp = self.df[list(available.keys())].copy().fillna(
            self.df[list(available.keys())].median()
        )
        scaled = pd.DataFrame(mm.fit_transform(df_comp), columns=df_comp.columns)

        for col, higher_is_better in available.items():
            if higher_is_better:
                components.append(scaled[col])
            else:
                components.append(1 - scaled[col])

        self.df["education_quality_index"] = pd.concat(components, axis=1).mean(axis=1).round(4)
        self._log("education_quality_index",
                  "Composite education quality index (0-1 scale) combining teacher quality, "
                  "infrastructure, completion, and dropout indicators")
        return self

    # ------------------------------------------------------------------
    # Lag of target (critical for forecasting)
    # ------------------------------------------------------------------
    def add_target_lags(self) -> "FeatureEngineer":
        """Add lag versions of the target variable for forecasting."""
        if self.TARGET not in self.df.columns:
            return self
        for lag in [1, 2]:
            new_name = f"csee_pass_rate_lag{lag}"
            if new_name not in self.df.columns:  # avoid overwrite
                self.df[new_name] = self.df.groupby("region")[self.TARGET].shift(lag)
                self._log(new_name, f"Lag-{lag} CSEE pass rate — captures system momentum")

        # Cumulative improvement trend
        self.df["csee_cumulative_improvement"] = (
            self.df.groupby("region")[self.TARGET]
            .transform(lambda s: s.diff().cumsum())
        )
        self._log("csee_cumulative_improvement",
                  "Cumulative change in CSEE pass rate since first available year")
        return self

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def build(self) -> pd.DataFrame:
        """
        Run the full feature engineering pipeline and return the enriched DataFrame.
        """
        (self
         .add_lag_features()
         .add_rolling_features()
         .add_growth_features()
         .add_infrastructure_index()
         .add_gender_features()
         .add_system_scale_features()
         .add_interaction_features()
         .add_education_quality_index()
         .add_target_lags())

        logger.info(f"Feature engineering complete: {len(self.df.columns)} total columns")
        return self.df

    def feature_descriptions(self) -> pd.DataFrame:
        """Return a DataFrame of all engineered features with descriptions."""
        return pd.DataFrame(self.feature_log)


# ---------------------------------------------------------------------------
# Model feature selector
# ---------------------------------------------------------------------------
def get_model_features(df: pd.DataFrame,
                       target: str = "csee_pass_rate",
                       include_lags: bool = True,
                       include_interactions: bool = True) -> List[str]:
    """
    Return the list of features to use for modelling.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered DataFrame.
    target : str
        Target column name (excluded from features).
    include_lags : bool
        Whether to include lag features.
    include_interactions : bool
        Whether to include interaction features.

    Returns
    -------
    List[str] of feature column names.
    """
    exclude_patterns = [
        target, "region", "year", "year_index",
        "infra_imputed_2020",
        # Leakage columns
        "csee_fail_rate",
        "enrolment_for_dropout",
    ]
    if not include_lags:
        exclude_patterns.extend(["_lag", "_roll"])
    if not include_interactions:
        exclude_patterns.extend(["_x_", "interaction"])

    features = []
    for col in df.columns:
        if any(pat in col for pat in exclude_patterns):
            continue
        if df[col].dtype in [float, int, np.float64, np.int64]:
            if df[col].notna().sum() > 10:
                features.append(col)
    return features
