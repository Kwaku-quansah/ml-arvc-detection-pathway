"""
Friedman + Nemenyi post-hoc analysis for model AUCs.

This script wraps the original notebook code that compared the eight models:

    - Gradient Boosted Trees
    - Random Forest
    - Decision Tree
    - Naive Bayes
    - OLS
    - Lasso Logistic Regression
    - Logistic Regression
    - TabNet

It:
- Runs the Friedman test on AUC values across folds.
- Computes average ranks and the Nemenyi Critical Difference (CD).
- Plots a simple CD diagram.
- Saves:
    - reports/tables/friedman_nemenyi_auc_ranks.csv
    - reports/figures/friedman_nemenyi_cd_diagram.svg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import friedmanchisquare
from pathlib import Path

from .config import REPORTS_DIR


def get_auc_dataframe() -> pd.DataFrame:
    """
    Return the DataFrame of per-fold AUC values for each model.

    This currently uses the hard-coded values from the original notebook.
    If you later want to load these from a CSV, you can modify this function.
    """
    data = {
        "Gradient Boosted Trees": [0.97, 0.90, 0.99, 0.96, 0.92],
        "Random Forest": [0.95, 0.90, 0.99, 0.94, 0.92],
        "Decision Tree": [0.91, 0.86, 0.97, 0.89, 0.90],
        "Naive Bayes": [0.95, 0.90, 0.97, 0.93, 0.92],
        "OLS": [0.95, 0.89, 0.97, 0.94, 0.92],
        "Lasso Logistic Regression": [0.94, 0.89, 0.96, 0.95, 0.93],
        "Logistic Regression": [0.96, 0.90, 0.97, 0.93, 0.93],
        "TabNet": [0.92, 0.87, 0.97, 0.91, 0.87],
    }
    df = pd.DataFrame(data)
    return df


def run_friedman_nemenyi():
    # Load AUCs
    df = get_auc_dataframe()

    # --- Friedman Test ---
    stat, p = friedmanchisquare(*[df[c] for c in df.columns])
    print(f"Friedman statistic = {stat:.4f}, p-value = {p:.6f}")

    # --- Average ranks ---
    ranks = df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()
    print("\nAverage ranks (lower = better):")
    print(avg_ranks)

    # --- Nemenyi critical difference ---
    k = len(avg_ranks)  # number of models
    n = len(df)         # number of folds / datasets
    CD = 2.728 * np.sqrt(k * (k + 1) / (6.0 * n))  # q_alpha for alpha=0.05, k→∞
    print(f"\nCritical Difference (α=0.05): {CD:.3f}")

    # Ensure output dirs exist
    tables_dir = REPORTS_DIR / "tables"
    figures_dir = REPORTS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save ranks + CD + p-value to CSV
    out_df = pd.DataFrame({"avg_rank": avg_ranks})
    out_df["model"] = out_df.index
    out_df["friedman_statistic"] = stat
    out_df["friedman_p_value"] = p
    out_df["nemenyi_CD_alpha_0.05"] = CD

    csv_path = tables_dir / "friedman_nemenyi_auc_ranks.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"\nRank summary saved to {csv_path}")

    # --- Plot: Critical Difference diagram (simple version) ---
    plt.figure(figsize=(10, 3))
    # baseline line
    plt.hlines(1, avg_ranks.min() - 0.5, avg_ranks.max() + 0.5, color="k", lw=1)
    # points
    plt.scatter(avg_ranks, np.ones_like(avg_ranks), s=80, zorder=3)

    # stagger labels a bit vertically
    y_offsets = np.linspace(-0.05, 0.05, len(avg_ranks))
    for (i, (model, rank)) in enumerate(avg_ranks.items()):
        plt.text(
            rank,
            1.05 + y_offsets[i],
            model,
            rotation=45,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    # CD bar from best rank
    best_rank = avg_ranks.min()
    plt.plot([best_rank, best_rank + CD], [1.2, 1.2], color="k", lw=2)
    plt.text(
        best_rank + CD / 2,
        1.23,
        f"CD = {CD:.2f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    plt.title("Critical Difference Diagram (Friedman + Nemenyi, AUC Ranks)", fontsize=12)
    plt.yticks([])
    plt.xlabel("Average Rank (Lower = Better)")
    plt.tight_layout()

    fig_path = figures_dir / "friedman_nemenyi_cd_diagram.svg"
    plt.savefig(fig_path, format="svg", bbox_inches="tight")
    plt.show()
    print(f"CD diagram saved to {fig_path}")


def main():
    run_friedman_nemenyi()


if __name__ == "__main__":
    main()

