"""
preprocessing.py
================
Data cleaning, validation, imputation, and preprocessing pipeline for
Tanzania BEST education panel data.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

logger = logging.getLogger("preprocessing")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_REGIONS = [
    "ARUSHA", "DAR ES SALAAM", "DODOMA", "GEITA", "IRINGA", "KAGERA",
    "KATAVI", "KIGOMA", "KILIMANJARO", "LINDI", "MANYARA", "MARA",
    "MBEYA", "MOROGORO", "MTWARA", "MWANZA", "NJOMBE", "PWANI",
    "RUKWA", "RUVUMA", "SHINYANGA", "SIMIYU", "SINGIDA", "SONGWE",
    "TABORA", "TANGA",
]

# Hard-coded fallback values (from BEST report notes)
FALLBACK_VALUES = {
    2020: {"ptr_national": 22.5, "dropout_rate_pct": 4.59,
           "gross_completion_rate": 38.55, "csee_pass_rate": 85.84,
           "csee_candidates": 435654.0},
    2021: {"ptr_national": 23.0, "dropout_rate_pct": 4.57,
           "gross_completion_rate": 39.51},
    2022: {"ptr_national": 24.0, "dropout_rate_pct": 4.48},
    2023: {"ptr_national": 25.0, "dropout_rate_pct": 4.31},
    2024: {"ptr_national": 26.0},
}

NUMERIC_COLS = [
    "govt_schools", "nongovt_schools", "total_schools", "enrolment_f1f4",
    "total_teachers", "schools_with_electricity", "pct_schools_electricity",
    "desktop_computers", "laptop_computers", "projectors",
    "enrolment_for_dropout", "total_dropouts", "dropout_rate_regional",
    "pupil_book_ratio", "disability_enrolment",
    "csee_pass_rate", "csee_fail_rate", "csee_div1_pct", "csee_candidates",
    "dropout_rate_pct", "gross_completion_rate",
    "total_teachers_national", "qualified_teachers_national", "ptr_national",
]


# ---------------------------------------------------------------------------
# Core cleaner
# ---------------------------------------------------------------------------
class BESTCleaner:
    """
    Clean, validate, and impute the raw BEST panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Raw panel from BESTLoader.build_regional_panel().
    log_transforms : bool
        If True, print a transformation log after each step.
    """

    def __init__(self, panel: pd.DataFrame, log_transforms: bool = True):
        self.panel = panel.copy()
        self.log_transforms = log_transforms
        self.transform_log: List[str] = []

    def _log(self, msg: str) -> None:
        self.transform_log.append(msg)
        if self.log_transforms:
            logger.info(msg)

    def run(self) -> pd.DataFrame:
        """Execute the full cleaning pipeline and return the cleaned panel."""
        self._drop_duplicates()
        self._coerce_numerics()
        self._apply_fallback_values()
        self._impute_regional_infrastructure()
        self._clip_impossible_values()
        self._derive_computed_columns()
        self._derive_qualified_ratio()
        self._filter_valid_regions()
        self._sort_panel()
        return self.panel

    # ------------------------------------------------------------------
    def _drop_duplicates(self) -> None:
        n0 = len(self.panel)
        self.panel = self.panel.drop_duplicates(subset=["year", "region"])
        n1 = len(self.panel)
        self._log(f"Duplicates removed: {n0 - n1} rows")

    def _coerce_numerics(self) -> None:
        for col in NUMERIC_COLS:
            if col in self.panel.columns:
                self.panel[col] = pd.to_numeric(self.panel[col], errors="coerce")
        self._log("Numeric coercion applied to all indicator columns")

    def _apply_fallback_values(self) -> None:
        """Apply fallback (hard-coded) values for known missing national series."""
        for yr, vals in FALLBACK_VALUES.items():
            for col, val in vals.items():
                if col in self.panel.columns:
                    mask = (self.panel["year"] == yr) & (self.panel[col].isna())
                    self.panel.loc[mask, col] = val
        self._log("Fallback values applied for missing national indicators")

    def _impute_regional_infrastructure(self) -> None:
        """
        Forward- and backward-fill regional infrastructure indicators
        (electricity, ICT) for the 2020 year, which lacks regional data.
        """
        self.panel = self.panel.sort_values(["region", "year"]).reset_index(drop=True)
        infra_cols = ["pct_schools_electricity", "desktop_computers",
                      "laptop_computers", "projectors", "pupil_book_ratio",
                      "disability_enrolment"]
        for col in infra_cols:
            if col in self.panel.columns:
                self.panel[col] = (
                    self.panel.groupby("region")[col]
                    .transform(lambda s: s.bfill().ffill())
                )
        # Mark imputed 2020 rows
        self.panel["infra_imputed_2020"] = (
            (self.panel["year"] == 2020) &
            self.panel["pct_schools_electricity"].notna()
        ).astype(int)
        self._log("Regional infrastructure imputed for 2020 via bfill/ffill")

    def _clip_impossible_values(self) -> None:
        """Clip known impossible values (negative counts, >100% rates, etc.)."""
        count_cols = [c for c in ["total_schools", "total_teachers", "enrolment_f1f4",
                                   "desktop_computers", "laptop_computers"] if c in self.panel.columns]
        for col in count_cols:
            neg_mask = self.panel[col] < 0
            if neg_mask.any():
                self.panel.loc[neg_mask, col] = np.nan
                self._log(f"Clipped {neg_mask.sum()} negative values in {col}")

        pct_cols = [c for c in ["pct_schools_electricity", "csee_pass_rate",
                                  "csee_fail_rate", "dropout_rate_pct",
                                  "gross_completion_rate"] if c in self.panel.columns]
        for col in pct_cols:
            mask = (self.panel[col] < 0) | (self.panel[col] > 100)
            if mask.any():
                self.panel.loc[mask, col] = np.nan
                self._log(f"Clipped {mask.sum()} out-of-range values in {col}")

    def _derive_computed_columns(self) -> None:
        """Derive basic computed columns from raw indicators."""
        if "enrolment_f1f4" in self.panel.columns and "total_teachers" in self.panel.columns:
            self.panel["ptr_regional"] = (
                self.panel["enrolment_f1f4"] / self.panel["total_teachers"]
            ).replace([np.inf, -np.inf], np.nan).round(2)

        if "desktop_computers" in self.panel.columns and "total_schools" in self.panel.columns:
            self.panel["desktops_per_school"] = (
                self.panel["desktop_computers"] / self.panel["total_schools"]
            ).replace([np.inf, -np.inf], np.nan).round(2)

        if "total_teachers" in self.panel.columns and "total_schools" in self.panel.columns:
            self.panel["teachers_per_school"] = (
                self.panel["total_teachers"] / self.panel["total_schools"]
            ).replace([np.inf, -np.inf], np.nan).round(2)

        if "nongovt_schools" in self.panel.columns and "total_schools" in self.panel.columns:
            self.panel["nongovt_share"] = (
                self.panel["nongovt_schools"] / self.panel["total_schools"]
            ).replace([np.inf, -np.inf], np.nan).round(4)

        if "enrolment_f1f4" in self.panel.columns and "total_schools" in self.panel.columns:
            self.panel["enrolment_per_school"] = (
                self.panel["enrolment_f1f4"] / self.panel["total_schools"]
            ).replace([np.inf, -np.inf], np.nan).round(1)

        if "laptop_computers" in self.panel.columns and "total_schools" in self.panel.columns:
            self.panel["laptops_per_school"] = (
                self.panel["laptop_computers"] / self.panel["total_schools"]
            ).replace([np.inf, -np.inf], np.nan).round(2)

        self._log("Basic computed columns derived")

    def _derive_qualified_ratio(self) -> None:
        """Derive the qualified teacher ratio from national totals."""
        if ("qualified_teachers_national" in self.panel.columns and
                "total_teachers_national" in self.panel.columns):
            self.panel["qualified_teacher_ratio"] = (
                self.panel["qualified_teachers_national"] /
                self.panel["total_teachers_national"]
            ).replace([np.inf, -np.inf], np.nan).round(4)
            self._log("Qualified teacher ratio derived")

    def _filter_valid_regions(self) -> None:
        n0 = len(self.panel)
        self.panel = self.panel[self.panel["region"].isin(EXPECTED_REGIONS)]
        self._log(f"Filtered to 26 expected regions: kept {len(self.panel)} of {n0} rows")

    def _sort_panel(self) -> None:
        self.panel = self.panel.sort_values(["region", "year"]).reset_index(drop=True)

    def missing_report(self) -> pd.DataFrame:
        """Return a summary of missing values per column."""
        miss = self.panel.isnull().sum().reset_index()
        miss.columns = ["column", "n_missing"]
        miss["pct_missing"] = (miss["n_missing"] / len(self.panel) * 100).round(1)
        return miss[miss["n_missing"] > 0].sort_values("pct_missing", ascending=False)

    def data_quality_report(self) -> pd.DataFrame:
        """Return a high-level data quality summary."""
        rows = []
        for col in self.panel.columns:
            if self.panel[col].dtype in [float, int]:
                rows.append({
                    "column": col,
                    "dtype": str(self.panel[col].dtype),
                    "n_missing": self.panel[col].isna().sum(),
                    "pct_missing": round(self.panel[col].isna().mean() * 100, 1),
                    "mean": round(self.panel[col].mean(), 3) if self.panel[col].notna().any() else np.nan,
                    "std": round(self.panel[col].std(), 3) if self.panel[col].notna().any() else np.nan,
                    "min": self.panel[col].min(),
                    "max": self.panel[col].max(),
                })
        return pd.DataFrame(rows)

    def get_transform_log(self) -> List[str]:
        """Return the transformation log."""
        return self.transform_log


# ---------------------------------------------------------------------------
# Scaler utility
# ---------------------------------------------------------------------------
def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                   method: str = "standard") -> Tuple[np.ndarray, np.ndarray, object]:
    """
    Scale features using the specified method.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    method  : str in {"standard", "minmax", "robust"}

    Returns
    -------
    (X_train_scaled, X_test_scaled, scaler)
    """
    scalers = {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
    }
    scaler = scalers.get(method, StandardScaler())
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)
    return X_tr_sc, X_te_sc, scaler


# ---------------------------------------------------------------------------
# Missing value summary helper
# ---------------------------------------------------------------------------
def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a neat missing-value summary DataFrame."""
    miss = df.isnull().sum().reset_index()
    miss.columns = ["column", "n_missing"]
    miss["pct_missing"] = (miss["n_missing"] / len(df) * 100).round(1)
    miss["data_type"] = miss["column"].map(df.dtypes.astype(str))
    return miss.sort_values("pct_missing", ascending=False).reset_index(drop=True)
