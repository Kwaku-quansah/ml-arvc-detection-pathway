"""
Pearson correlation analysis of ARVC multimodal features.

This is the structured version of your notebook code that did:

    os.chdir('R:/arvc_ml')
    df = pd.read_csv('dataset_knn_imputed_1.csv')
    df = df.drop(columns=['ARVC diagnosed'])
    corr_matrix = df.corr(method='pearson')
    ...

Here we:
- Use `load_knn_imputed_multimodal()` from data.py to get features + labels.
- Compute a Pearson correlation matrix across features.
- Save:
    * reports/tables/correlation_matrix_pearson.csv
    * reports/figures/correlation_matrix_pearson.svg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data import load_knn_imputed_multimodal
from .config import REPORTS_DIR


def compute_and_save_correlation_matrix():
    # Load standardized features, labels, and feature names
    # NOTE: Pearson correlations are invariant to linear scaling, so using
    # standardized X_scaled is fine (equivalent to your original code).
    X_scaled, y, feature_names = load_knn_imputed_multimodal()

    # Rebuild DataFrame similar to your original `df` (without the target)
    df_features = pd.DataFrame(X_scaled, columns=feature_names)

    # Calculate the Pearson correlation matrix
    corr_matrix = df_features.corr(method="pearson")

    # Ensure output directories exist
    tables_dir = REPORTS_DIR / "tables"
    figures_dir = REPORTS_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save correlation matrix to CSV
    csv_path = tables_dir / "correlation_matrix_pearson.csv"
    corr_matrix.to_csv(csv_path)
    print(f"Pearson correlation matrix saved to {csv_path}")

    # Plot the correlation matrix using matplotlib (your original style)
    plt.figure(figsize=(10, 6))

    # Display the correlation matrix as an image
    im = plt.imshow(corr_matrix, interpolation="nearest", aspect="auto")
    plt.colorbar(im)

    # Set the ticks and labels for the axes (feature names)
    plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
    plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

    # Add title to the plot
    plt.title("Pearson Correlation Matrix of Features")

    plt.tight_layout()
    fig_path = figures_dir / "pearson_correlation_matrix_of_features.svg"
    plt.savefig(fig_path, format="svg", bbox_inches="tight")
    plt.show()

    print(f"Pearson correlation heatmap saved to {fig_path}")


def main():
    compute_and_save_correlation_matrix()


if __name__ == "__main__":
    main()

