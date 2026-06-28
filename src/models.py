"""
models.py
=========
Machine learning model training, hyperparameter optimisation, and comparison
for Tanzania BEST education performance prediction.

Implements 10+ regression models with unified training, CV, and evaluation interface.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    HistGradientBoostingRegressor, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import (
    GridSearchCV, KFold, LeaveOneGroupOut,
    RandomizedSearchCV, RepeatedKFold, TimeSeriesSplit,
    cross_val_score,
)
from sklearn.metrics import (
    explained_variance_score, mean_absolute_error,
    mean_absolute_percentage_error, mean_squared_error,
    median_absolute_error, r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger("models")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def get_model_registry() -> Dict[str, BaseEstimator]:
    """Return a dictionary of all regression models to compare."""
    return {
        "Baseline (Mean)":          DummyRegressor(strategy="mean"),
        "Baseline (Median)":        DummyRegressor(strategy="median"),
        "Linear Regression":        LinearRegression(),
        "Ridge":                    Ridge(alpha=1.0),
        "Lasso":                    Lasso(alpha=0.1, max_iter=10000),
        "ElasticNet":               ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000),
        "Decision Tree":            DecisionTreeRegressor(max_depth=4, min_samples_leaf=3, random_state=42),
        "Random Forest":            RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42),
        "Extra Trees":              ExtraTreesRegressor(n_estimators=200, max_depth=5, min_samples_leaf=3, random_state=42),
        "Gradient Boosting":        GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, min_samples_leaf=3, random_state=42),
        "HistGradientBoosting":     HistGradientBoostingRegressor(max_iter=150, max_depth=4, learning_rate=0.05, random_state=42),
        "AdaBoost":                 AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    n_features: int = 1) -> Dict[str, float]:
    """Compute a comprehensive set of regression metrics."""
    n = len(y_true)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - n_features - 1, 1)
    mae   = mean_absolute_error(y_true, y_pred)
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    med_ae = median_absolute_error(y_true, y_pred)
    ev    = explained_variance_score(y_true, y_pred)
    # MAPE — guard against zero true values
    mask = y_true != 0
    mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100 if mask.any() else np.nan
    return {
        "R2":          round(r2, 4),
        "Adj_R2":      round(adj_r2, 4),
        "MAE":         round(mae, 4),
        "MSE":         round(mse, 4),
        "RMSE":        round(rmse, 4),
        "MedAE":       round(med_ae, 4),
        "MAPE_pct":    round(mape, 2) if not np.isnan(mape) else np.nan,
        "ExplVar":     round(ev, 4),
    }


# ---------------------------------------------------------------------------
# Cross-validation evaluator
# ---------------------------------------------------------------------------
class ModelEvaluator:
    """
    Train and evaluate multiple regression models using various CV strategies.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray or pd.Series
    groups  : array-like, optional
        Group labels for LeaveOneGroupOut CV (typically year values).
    n_jobs : int
        Parallel jobs for CV.
    """

    def __init__(self,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 groups: Optional[np.ndarray] = None,
                 n_jobs: int = -1):
        self.X_train = X_train
        self.y_train = np.asarray(y_train)
        self.groups  = groups
        self.n_jobs  = n_jobs
        self.results: List[Dict] = []
        self.trained_models: Dict[str, BaseEstimator] = {}

    def evaluate_all(self, models: Optional[Dict[str, BaseEstimator]] = None,
                     cv_strategy: str = "logo") -> pd.DataFrame:
        """
        Train and cross-validate all models.

        Parameters
        ----------
        models : dict, optional
            Model name -> estimator mapping. Defaults to get_model_registry().
        cv_strategy : str
            One of "kfold", "repeated_kfold", "logo", "timeseries".

        Returns
        -------
        pd.DataFrame with sorted comparison results.
        """
        if models is None:
            models = get_model_registry()

        cv = self._get_cv(cv_strategy)
        results = []

        for name, model in models.items():
            t0 = time.time()
            try:
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv,
                    groups=self.groups if cv_strategy == "logo" else None,
                    scoring="neg_mean_absolute_error",
                    n_jobs=self.n_jobs,
                )
                cv_r2 = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv,
                    groups=self.groups if cv_strategy == "logo" else None,
                    scoring="r2",
                    n_jobs=self.n_jobs,
                )
                # Fit on full training data
                model.fit(self.X_train, self.y_train)
                self.trained_models[name] = model

                elapsed = round(time.time() - t0, 2)
                results.append({
                    "Model":        name,
                    "CV_MAE_mean":  round(-cv_scores.mean(), 4),
                    "CV_MAE_std":   round(cv_scores.std(), 4),
                    "CV_R2_mean":   round(cv_r2.mean(), 4),
                    "CV_R2_std":    round(cv_r2.std(), 4),
                    "CV_Strategy":  cv_strategy,
                    "Fit_Time_s":   elapsed,
                })
                logger.info(f"  {name}: CV_MAE={-cv_scores.mean():.4f} "
                            f"R2={cv_r2.mean():.4f} [{elapsed}s]")
            except Exception as e:
                logger.error(f"  {name} failed: {e}")
                results.append({"Model": name, "CV_MAE_mean": np.nan,
                                "CV_R2_mean": np.nan, "error": str(e)})

        self.results = results
        df = pd.DataFrame(results).sort_values("CV_MAE_mean").reset_index(drop=True)
        df.index = df.index + 1  # 1-indexed rank
        return df

    def evaluate_test(self, X_test: np.ndarray, y_test: np.ndarray,
                      n_features: int = 1) -> pd.DataFrame:
        """Evaluate all trained models on the held-out test set."""
        rows = []
        for name, model in self.trained_models.items():
            preds = model.predict(X_test)
            metrics = compute_metrics(y_test, preds, n_features=n_features)
            metrics["Model"] = name
            rows.append(metrics)
        df = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
        df.index = df.index + 1
        return df

    def _get_cv(self, strategy: str):
        """Build the cross-validation object."""
        if strategy == "logo" and self.groups is not None:
            return LeaveOneGroupOut()
        elif strategy == "repeated_kfold":
            return RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)
        elif strategy == "timeseries":
            return TimeSeriesSplit(n_splits=4)
        else:
            return KFold(n_splits=5, shuffle=True, random_state=42)


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
PARAM_GRIDS: Dict[str, Dict] = {
    "Random Forest": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 4, 5, None],
        "min_samples_leaf": [2, 3, 5],
        "max_features":     ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [2, 3, 4],
        "learning_rate":    [0.02, 0.05, 0.1],
        "min_samples_leaf": [2, 3, 5],
        "subsample":        [0.8, 1.0],
    },
    "Extra Trees": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 4, 5, None],
        "min_samples_leaf": [2, 3, 5],
    },
    "HistGradientBoosting": {
        "max_iter":      [100, 200],
        "max_depth":     [3, 4, 5],
        "learning_rate": [0.02, 0.05, 0.1],
    },
    "Ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "Lasso": {
        "alpha": [0.001, 0.01, 0.1, 1.0],
    },
    "ElasticNet": {
        "alpha":    [0.01, 0.1, 1.0],
        "l1_ratio": [0.2, 0.5, 0.8],
    },
}


def tune_model(model: BaseEstimator, model_name: str,
               X_train: np.ndarray, y_train: np.ndarray,
               groups: Optional[np.ndarray] = None,
               method: str = "grid",
               n_iter: int = 50,
               cv_strategy: str = "logo") -> Tuple[BaseEstimator, pd.DataFrame]:
    """
    Tune a single model using GridSearchCV or RandomizedSearchCV.

    Parameters
    ----------
    model : BaseEstimator
    model_name : str
        Must match a key in PARAM_GRIDS.
    X_train, y_train : arrays
    groups : array-like, optional
        For LOGO cross-validation.
    method : str
        "grid" or "random".
    n_iter : int
        Iterations for RandomizedSearchCV.
    cv_strategy : str

    Returns
    -------
    (best_estimator, cv_results_df)
    """
    param_grid = PARAM_GRIDS.get(model_name, {})
    if not param_grid:
        logger.warning(f"No parameter grid for {model_name}; returning as-is.")
        model.fit(X_train, y_train)
        return model, pd.DataFrame()

    if cv_strategy == "logo" and groups is not None:
        cv = LeaveOneGroupOut()
        cv_splits = list(cv.split(X_train, y_train, groups))
    else:
        cv_splits = KFold(n_splits=5, shuffle=True, random_state=42)

    if method == "random":
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv_splits,
            scoring="neg_mean_absolute_error", n_jobs=-1,
            random_state=42, verbose=0,
        )
    else:
        search = GridSearchCV(
            model, param_grid, cv=cv_splits,
            scoring="neg_mean_absolute_error", n_jobs=-1,
            verbose=0,
        )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    logger.info(f"Tuning {model_name}: best MAE = {-search.best_score_:.4f} "
                f"in {elapsed}s")
    logger.info(f"Best params: {search.best_params_}")

    cv_results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")

    return search.best_estimator_, cv_results
