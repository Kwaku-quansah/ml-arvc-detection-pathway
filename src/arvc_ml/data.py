"""
Data loading utilities for ARVC ML models.

⚠️ IMPORTANT
-----------
This project does NOT ship any clinical data.

The functions below are *stubs* that define the public interface used by the
model scripts (e.g., gbt.py, random_forest.py, etc.). To actually run the
models, you must implement these functions locally to load your own
(pre-imputed, de-identified) datasets.

By default they raise a RuntimeError so that:
- The repository is safe to share publicly (no PHI / clinical CSVs).
- You still get a clear message telling you what to do on your local machine.
"""

from typing import Tuple, List

import numpy as np


def load_knn_imputed_multimodal() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the KN-imputed multimodal feature matrix and labels.

    Expected return:
        X_scaled: (n_samples, n_features) NumPy array of standardized features
        y:        (n_samples,) NumPy array of binary labels (0/1 for ARVC)
        feature_names: list of column names corresponding to X_scaled columns

    You should implement this locally something like:

        import pandas as pd
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv("/path/to/dataset_knn_imputed_1.csv")
        y = df["ARVC diagnosed"].to_numpy()
        X = df.drop(columns=["ARVC diagnosed"])
        feature_names = list(X.columns)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.to_numpy())

        return X_scaled, y, feature_names
    """
    raise RuntimeError(
        "load_knn_imputed_multimodal() is a stub. "
        "Implement it locally to load and scale your multimodal clinical dataset."
    )


def load_knn_imputed_ecg_only() -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load the KN-imputed ECG-only feature matrix and labels.

    Expected return:
        X_scaled: (n_samples, n_features) NumPy array of standardized features
        y:        (n_samples,) NumPy array of binary labels (0/1 for ARVC)
        feature_names: list of column names corresponding to X_scaled columns

    Locally, you might do something like:

        df = pd.read_csv("/path/to/dataset_knn_imputed_1.csv")

        ecg_cols = [
            # e.g. 'QRSDuration', 'PVC_count', 'TWI_V2', ...
        ]
        X = df[ecg_cols]
        y = df["ARVC diagnosed"].to_numpy()
        feature_names = ecg_cols

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.to_numpy())

        return X_scaled, y, feature_names
    """
    raise RuntimeError(
        "load_knn_imputed_ecg_only() is a stub. "
        "Implement it locally to load and scale your ECG-only dataset."
    )
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import PROJECT_ROOT, TARGET_COL

def load_knn_imputed_multimodal():
    csv_path = PROJECT_ROOT / "dataset_knn_imputed_1.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found.\nPlace your private dataset in the repo root (not tracked in git)."
        )

    df = pd.read_csv(csv_path)

    X = df.drop(columns=[TARGET_COL]).to_numpy()
    y = df[TARGET_COL].to_numpy()
    feature_names = list(df.drop(columns=[TARGET_COL]).columns)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names

