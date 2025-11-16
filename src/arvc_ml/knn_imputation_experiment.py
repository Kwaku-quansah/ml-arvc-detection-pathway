"""
KNN imputation experiment for ARVC dataset.

This module wraps the original notebook KNN evaluation code into a reusable
pipeline component.

It:
- Loads the incomplete dataset via `load_raw_for_imputation()` (stub in data.py).
- Evaluates KNN imputation for k in [1, 3, 5, 7, 9] using artificial masking.
- Computes:
    * Mean/Std/Skew discrepancies
    * Correlation discrepancies
    * Distribution discrepancies (Wasserstein distance)
    * Reconstruction RMSE on masked entries
- Normalizes and combines these into a composite score.
- Saves:
    * A metrics CSV  -> reports/tables/knn_imputation_metrics.csv
    * A plot         -> reports/figures/knn_imputation_quality_vs_k.svg

NOTE: This does NOT ship any clinical data. You must implement
`load_raw_for_imputation()` locally with your own dataset.
"""

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance

from .data import load_raw_for_imputation
from .config import REPORTS_DIR


def evaluate_knn_on_incomplete_data(
    df_incomplete: pd.DataFrame,
    ks: Sequence[int] = (1, 3, 5, 7, 9),
    mask_frac: float = 0.1,
    random_state: int = 42,
    plot_path: str | None = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Evaluate KNN imputation on incomplete dataset.

    Uses artificial masking of known values to estimate imputation accuracy
    and distribution preservation.

    Parameters
    ----------
    df_incomplete : pd.DataFrame
        Dataset with missing values.
    ks : sequence of int
        List of K values to test.
    mask_frac : float
        Fraction of known (non-missing) entries to hide for validation.
    random_state : int
        Random seed for reproducibility.
    plot_path : str or Path, optional
        If provided, a line plot of metrics vs K is saved here.

    Returns
    -------
    results_df : pd.DataFrame
        Evaluation metrics per K (indexed by K).
    best_k : int
        K value minimizing the composite discrepancy score.
    """

    rng = np.random.default_rng(random_state)

    # Mask additional known values (only from non-missing positions)
    not_nan_mask = ~df_incomplete.isna()
    mask_candidates = np.where(not_nan_mask)
    n_mask = int(mask_frac * len(mask_candidates[0]))
    selected = rng.choice(len(mask_candidates[0]), n_mask, replace=False)

    masked_df = df_incomplete.copy()
    mask_rows, mask_cols = mask_candidates[0][selected], mask_candidates[1][selected]
    mask_validation = np.zeros_like(df_incomplete, dtype=bool)
    mask_validation[mask_rows, mask_cols] = True
    masked_df.values[mask_validation] = np.nan

    # Helper metrics
    def compare_stats(original: pd.DataFrame, imputed: pd.DataFrame) -> float:
        stats = pd.DataFrame(
            {
                "mean_diff": imputed.mean() - original.mean(),
                "std_diff": imputed.std() - original.std(),
                "skew_diff": imputed.skew() - original.skew(),
            }
        )
        # Average absolute difference across all stats and columns
        return stats.abs().mean().mean()

    def corr_diff(original: pd.DataFrame, imputed: pd.DataFrame) -> float:
        corr_orig = original.corr()
        corr_imp = imputed.corr()
        return np.mean(np.abs(corr_orig - corr_imp))

    def dist_similarity(original: pd.DataFrame, imputed: pd.DataFrame) -> float:
        # Lower is better (smaller Wasserstein distance)
        return np.mean(
            [
                wasserstein_distance(
                    original[col].dropna(), imputed[col].dropna()
                )
                for col in original.columns
            ]
        )

    def reconstruction_rmse(
        original: pd.DataFrame, imputed: pd.DataFrame, mask: np.ndarray
    ) -> float:
        return np.sqrt(
            mean_squared_error(
                original.values[mask],
                imputed.values[mask],
            )
        )

    # Store results
    results: list[dict] = []

    for k in ks:
        imputer = KNNImputer(n_neighbors=int(k))
        df_imputed = pd.DataFrame(
            imputer.fit_transform(masked_df),
            columns=df_incomplete.columns,
            index=df_incomplete.index,
        )

        metrics = {
            "K": k,
            "Mean/Std/Skew Δ": compare_stats(df_incomplete, df_imputed),
            "Correlation Δ": corr_diff(df_incomplete, df_imputed),
            "Distribution Δ": dist_similarity(df_incomplete, df_imputed),
            "Reconstruction RMSE": reconstruction_rmse(
                df_incomplete, df_imputed, mask_validation
            ),
        }
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("K")

    # Normalize and combine metrics
    normed = (results_df - results_df.min()) / (results_df.max() - results_df.min())
    results_df["Composite Score"] = normed.mean(axis=1)
    best_k = results_df["Composite Score"].idxmin()

    # Plot metric trends (your original plotting logic)
    ax = results_df[
        ["Mean/Std/Skew Δ", "Correlation Δ", "Distribution Δ", "Reconstruction RMSE"]
    ].plot(marker="o", figsize=(10, 6))
    ax.set_title("KNN Imputation Quality vs. K (on Incomplete Data)")
    ax.set_ylabel("Metric (lower = better)")
    ax.set_xlabel("K")
    ax.grid(True)

    if plot_path is not None:
        plt.savefig(plot_path, bbox_inches="tight", dpi=300)
        print(f"KNN metric plot saved to {plot_path}")

    plt.show()

    return results_df, best_k


def run_knn_imputation_experiment(
    ks: Sequence[int] = (1, 3, 5, 7, 9),
    mask_frac: float = 0.1,
    random_state: int = 42,
) -> None:
    """
    High-level wrapper:
    - Loads incomplete dataset
    - Runs evaluation
    - Saves CSV + plot to `reports/`
    """
    # Load raw incomplete dataset (user implements locally)
    df_raw = load_raw_for_imputation()

    # Typically you’ll want to restrict to numeric columns for imputation stats
    df_numeric = df_raw.select_dtypes(include=["number"])
    if df_numeric.empty:
        raise ValueError(
            "No numeric columns found in the raw dataset; "
            "check load_raw_for_imputation()."
        )

    print(f"Evaluating KNN on shape: {df_numeric.shape}")

    # Ensure output dirs exist
    tables_dir = REPORTS_DIR / "tables"
    figures_dir = REPORTS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_path = figures_dir / "knn_imputation_quality_vs_k.svg"

    results_df, best_k = evaluate_knn_on_incomplete_data(
        df_incomplete=df_numeric,
        ks=ks,
        mask_frac=mask_frac,
        random_state=random_state,
        plot_path=str(plot_path),
    )

    csv_path = tables_dir / "knn_imputation_metrics.csv"
    results_df.to_csv(csv_path)
    print(f"KNN imputation metrics saved to {csv_path}")
    print(f"Best K according to composite score: {best_k}")


def main():
    run_knn_imputation_experiment()


if __name__ == "__main__":
    main()

