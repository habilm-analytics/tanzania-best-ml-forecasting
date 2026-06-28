"""
forecasting.py
==============
Forecasts key education indicators for 2025-2030 using multiple approaches:
ML-based recursive forecasting, ARIMA-style trend extrapolation, and
scenario-based modelling.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger("forecasting")


# ---------------------------------------------------------------------------
# Trend extrapolation (ARIMA-lite using polynomial fit)
# ---------------------------------------------------------------------------
def trend_forecast(series: pd.Series, horizon: int = 6,
                   degree: int = 2) -> pd.DataFrame:
    """
    Fit a polynomial trend to a time series and extrapolate.

    Parameters
    ----------
    series : pd.Series with integer year index.
    horizon : int
        Number of years to forecast.
    degree : int
        Polynomial degree (1=linear, 2=quadratic).

    Returns
    -------
    pd.DataFrame with columns: year, forecast, lower_ci, upper_ci.
    """
    years_obs = series.index.astype(float).values
    y_obs     = series.values.astype(float)

    # Fit polynomial
    coeffs  = np.polyfit(years_obs, y_obs, degree)
    poly_fn = np.poly1d(coeffs)

    # Residual std for CI
    residuals = y_obs - poly_fn(years_obs)
    res_std   = residuals.std()

    last_yr  = int(years_obs.max())
    fcast_yr = np.arange(last_yr + 1, last_yr + horizon + 1)
    fcast    = poly_fn(fcast_yr)

    # Cap at reasonable bounds (0-100 for rates)
    if series.max() <= 100:
        fcast = np.clip(fcast, 0, 100)

    df = pd.DataFrame({
        "year":      fcast_yr,
        "forecast":  fcast,
        "lower_ci":  np.clip(fcast - 1.96 * res_std, 0, fcast.max() + 5),
        "upper_ci":  np.clip(fcast + 1.96 * res_std, fcast.min() - 5, 100 if series.max() <= 100 else None),
        "method":    "Polynomial Trend",
    })
    return df


# ---------------------------------------------------------------------------
# ML recursive forecasting
# ---------------------------------------------------------------------------
class RecursiveForecaster:
    """
    Recursive (multi-step) ML forecaster for education indicators.

    Uses lagged features to generate one-year-ahead predictions,
    then feeds predictions back as inputs for subsequent years.

    Parameters
    ----------
    model : BaseEstimator
        A fitted or unfitted sklearn regressor.
    target : str
        Column name of the target indicator.
    feature_cols : list of str
        Feature columns to use (must be available in the training data).
    lags : list of int
        Lag values to use (default [1, 2]).
    """

    def __init__(self,
                 model: Optional[BaseEstimator] = None,
                 target: str = "csee_pass_rate",
                 feature_cols: Optional[List[str]] = None,
                 lags: List[int] = [1, 2]):
        self.model    = model or GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        self.target   = target
        self.lags     = lags
        self.feature_cols = feature_cols or []
        self.is_fitted = False
        self._training_cols: List[str] = []

    def _add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values("year").copy()
        for lag in self.lags:
            df[f"{self.target}_lag{lag}"] = df[self.target].shift(lag)
        return df

    def fit(self, df: pd.DataFrame) -> "RecursiveForecaster":
        """Fit the model on the national time series."""
        df = self._add_lags(df).dropna()
        lag_cols = [f"{self.target}_lag{lag}" for lag in self.lags]
        available_feats = [c for c in self.feature_cols if c in df.columns]
        use_cols = lag_cols + available_feats
        use_cols = [c for c in use_cols if c in df.columns]

        self._training_cols = use_cols
        X = df[use_cols].values
        y = df[self.target].values

        self.model.fit(X, y)
        self.is_fitted = True
        self._last_df = df  # retain for recursive step
        logger.info(f"RecursiveForecaster fitted on {len(df)} obs with {len(use_cols)} features")
        return self

    def forecast(self, horizon: int = 6) -> pd.DataFrame:
        """Forecast the next `horizon` years."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        # Start from the last known values
        history = self._last_df[["year", self.target]].copy()
        last_yr = int(history["year"].max())

        forecasts = []
        current_series = history.set_index("year")[self.target].to_dict()

        for step in range(1, horizon + 1):
            pred_yr = last_yr + step
            row: Dict = {}

            for lag in self.lags:
                lag_yr = pred_yr - lag
                row[f"{self.target}_lag{lag}"] = current_series.get(lag_yr, np.nan)

            # Additional features — use last known values
            for feat in self.feature_cols:
                if feat in self._training_cols and feat in self._last_df.columns:
                    row[feat] = self._last_df[feat].iloc[-1]

            x_row = np.array([row.get(c, np.nan) for c in self._training_cols]).reshape(1, -1)

            if np.isnan(x_row).any():
                # Fill NaNs with column means from training
                logger.debug(f"NaN inputs at step {step}; filling with training means")
                X_train_arr = self._last_df[self._training_cols].values
                col_means = np.nanmean(X_train_arr, axis=0)
                nan_idx = np.isnan(x_row[0])
                x_row[0][nan_idx] = col_means[nan_idx]

            pred = float(self.model.predict(x_row)[0])

            # Clip to plausible range
            if self.target.endswith("_rate") or "pass" in self.target:
                pred = np.clip(pred, 0, 100)

            current_series[pred_yr] = pred
            forecasts.append({"year": pred_yr, "forecast": pred, "method": "ML Recursive"})

        return pd.DataFrame(forecasts)

    def get_cv_mae(self, df: pd.DataFrame) -> float:
        """Return MAE from leave-last-year-out validation."""
        if len(df) < 4:
            return np.nan
        df = self._add_lags(df).dropna().sort_values("year")
        lag_cols = [f"{self.target}_lag{lag}" for lag in self.lags]
        use_cols = [c for c in lag_cols if c in df.columns]
        X = df[use_cols].values
        y = df[self.target].values
        train_x, test_x = X[:-1], X[[-1]]
        train_y, test_y = y[:-1], y[[-1]]
        m = self.model.__class__(**self.model.get_params())
        m.fit(train_x, train_y)
        return float(mean_absolute_error(test_y, m.predict(test_x)))


# ---------------------------------------------------------------------------
# Scenario modeller
# ---------------------------------------------------------------------------
def scenario_forecast(
    baseline_series: pd.Series,
    model: BaseEstimator,
    horizon: int = 6,
    scenarios: Optional[Dict[str, Dict]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Generate optimistic, baseline, and pessimistic scenario forecasts.

    Parameters
    ----------
    baseline_series : pd.Series
        Historical national pass rate indexed by year.
    model : BaseEstimator
        Fitted single-feature (year-only) or multi-feature model.
    horizon : int
    scenarios : dict, optional
        Custom scenario definitions: {"label": {"annual_increment": float}}

    Returns
    -------
    dict of {scenario_label: DataFrame with year, forecast columns}
    """
    if scenarios is None:
        scenarios = {
            "Optimistic (+1.5%/yr)":  {"annual_increment": 1.5},
            "Baseline (trend)":       {"annual_increment": 0.8},
            "Pessimistic (+0.2%/yr)": {"annual_increment": 0.2},
        }

    last_yr  = int(baseline_series.index.max())
    last_val = float(baseline_series.iloc[-1])
    results: Dict[str, pd.DataFrame] = {}

    for label, params in scenarios.items():
        inc  = params.get("annual_increment", 0.5)
        yrs  = np.arange(last_yr + 1, last_yr + horizon + 1)
        vals = np.array([min(100, last_val + inc * step) for step in range(1, horizon + 1)])
        results[label] = pd.DataFrame({"year": yrs, "forecast": vals, "scenario": label})

    return results


# ---------------------------------------------------------------------------
# Forecast visualisation
# ---------------------------------------------------------------------------
def plot_forecast(
    historical: pd.Series,
    forecasts: Dict[str, pd.DataFrame],
    target_label: str = "CSEE Pass Rate (%)",
    title: str = "Tanzania CSEE Pass Rate Forecast (2025-2030)",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot historical data + multiple forecast scenarios."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Historical
    ax.plot(historical.index, historical.values, "o-", color="steelblue",
            lw=2.5, ms=7, label="Historical", zorder=5)

    colours = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for i, (label, df) in enumerate(forecasts.items()):
        colour = colours[i % len(colours)]
        ax.plot(df["year"], df["forecast"], "--", color=colour,
                lw=2, ms=6, marker="s", alpha=0.85, label=label)
        if "lower_ci" in df.columns and "upper_ci" in df.columns:
            ax.fill_between(df["year"], df["lower_ci"], df["upper_ci"],
                            alpha=0.15, color=colour)

    # Boundary line between historical and forecast
    last_hist_yr = int(historical.index.max())
    ax.axvline(last_hist_yr + 0.5, color="gray", lw=1.5,
               linestyle=":", label="Forecast start")
    ax.annotate("Forecast →", xy=(last_hist_yr + 0.6, historical.min() + 2),
                fontsize=9, color="gray")

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel(target_label, fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(historical.min() - 5, min(100, historical.max() + 15))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_regional_forecast_heatmap(
    panel: pd.DataFrame,
    forecast_years: List[int],
    target_col: str = "csee_pass_rate",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap showing actual values across regions and years,
    extended with forecast columns.
    """
    pivot = panel.pivot_table(index="region", columns="year",
                               values=target_col, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(14, 10))
    sns_ax = sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", annot=True, fmt=".1f",
        linewidths=0.4, linecolor="white",
        cbar_kws={"label": target_col},
    )
    ax.set_title(f"Regional {target_col} Heatmap (Actual: 2020-2024)",
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Region")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


import seaborn as sns
