"""
utilities.py
============
General utility functions: path management, logging setup, export helpers,
report generation, and reproducibility tools.

Author : Habil Masawika
Project: Forecasting Education Performance Using Tanzania BEST Data (2020-2024)
"""

import hashlib
import json
import logging
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(log_level: str = "INFO",
                  log_file: Optional[str] = None) -> logging.Logger:
    """Configure project-wide logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("BEST-ML")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds(seed: int = 42) -> None:
    """Fix random seeds for NumPy, Python random, and optionally TF/torch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_environment_info() -> Dict[str, str]:
    """Return a dictionary of key environment versions."""
    import sklearn
    import matplotlib
    import seaborn
    info = {
        "python":      sys.version.split()[0],
        "numpy":       np.__version__,
        "pandas":      pd.__version__,
        "sklearn":     sklearn.__version__,
        "matplotlib":  matplotlib.__version__,
        "seaborn":     seaborn.__version__,
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import joblib
        info["joblib"] = joblib.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Path management
# ---------------------------------------------------------------------------
class ProjectPaths:
    """Centralised path registry for the project."""

    def __init__(self, root: Optional[str] = None):
        if root is not None:
            self.root = Path(root)
        else:
            # Auto-detect: go up from src/ to project root
            self.root = Path(__file__).resolve().parent.parent
        self.data_raw     = self.root / "data" / "raw"
        self.data_proc    = self.root / "data" / "processed"
        self.notebooks    = self.root / "notebooks"
        self.src          = self.root / "src"
        self.models_dir   = self.root / "models"
        self.outputs      = self.root / "outputs"
        self.figures      = self.outputs / "figures"
        self.reports      = self.outputs / "reports"
        self.tables       = self.outputs / "tables"
        self.dashboard    = self.outputs / "dashboard"
        self.app          = self.root / "app"
        self._create_all()

    def _create_all(self) -> None:
        for attr, path in self.__dict__.items():
            if isinstance(path, Path):
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except PermissionError:
                    pass  # Read-only filesystem on Streamlit Cloud — skip

    def fig(self, name: str) -> str:
        """Return string path to a figure file."""
        return str(self.figures / name)

    def table(self, name: str) -> str:
        return str(self.tables / name)

    def model(self, name: str) -> str:
        return str(self.models_dir / name)

    def report(self, name: str) -> str:
        return str(self.reports / name)

    def processed(self, name: str) -> str:
        return str(self.data_proc / name)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------
def save_dataframe(df: pd.DataFrame, path: str,
                   index: bool = False) -> None:
    """Save a DataFrame to CSV, creating parent directories."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logging.getLogger("utilities").info(f"Saved DataFrame ({df.shape}) → {path}")


def save_model(model: Any, path: str) -> None:
    """Serialise a fitted model with joblib."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logging.getLogger("utilities").info(f"Saved model → {path}")


def load_model(path: str) -> Any:
    """Load a serialised model."""
    return joblib.load(path)


def save_json(data: Dict, path: str) -> None:
    """Save a dictionary as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Dict:
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------
def generate_text_report(sections: Dict[str, str], path: str) -> None:
    """Write a plain-text report from a dict of {section_title: content}."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("TANZANIA BEST EDUCATION ML PROJECT — ANALYSIS REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        for title, content in sections.items():
            f.write(f"{'─' * 60}\n")
            f.write(f"{title.upper()}\n")
            f.write(f"{'─' * 60}\n")
            f.write(content.strip())
            f.write("\n\n")
    logging.getLogger("utilities").info(f"Text report written → {path}")


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Convert a DataFrame to a Markdown table string."""
    try:
        return df.to_markdown(floatfmt=floatfmt, index=False)
    except Exception:
        return df.to_string(index=False)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------
class Timer:
    """Simple wall-clock timer."""

    def __init__(self, label: str = ""):
        self.label = label

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self._t0
        logging.getLogger("utilities").info(
            f"{self.label} completed in {elapsed:.2f}s"
        )
        self.elapsed = elapsed


# ---------------------------------------------------------------------------
# Data checksums
# ---------------------------------------------------------------------------
def dataframe_checksum(df: pd.DataFrame) -> str:
    """Return a SHA-256 hash of a DataFrame for reproducibility checks."""
    h = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()
    return h
