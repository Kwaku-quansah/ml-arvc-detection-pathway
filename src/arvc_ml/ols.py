"""
Ordinary Least Squares (OLS) multimodal model for ARVC.

- Uses LinearRegression for binary classification (continuous scores)
- 5-fold Stratified CV
- ROC curves per fold + mean ROC with 95% CI
- Confusion matrix, accuracy, sensitivity, specificity + 95% CIs

Data loading and scaling is handled by load_knn_imputed_multimodal() in data.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

from .data import load_knn_imputed_multimodal
from .config import REPORTS_DIR, RANDOM_STATE


def run_ols_cv(
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    output_prefix: str = "ols",
):
    """
    Run stratified K-fold CV with OLS (LinearRegression) and plot ROC curves.
    """

    # 🔁 Replaces df = pd.read_csv(...), data/target, X, y, scaler, etc.
    X_scaled, y, feature_names = load_knn_imputed_multimodal()

    # Containers for ROC and AUC across folds
    all_tpr_ols = []
    auc_scores_ols = []
    mean_fpr = np.linspace(0, 1, 100)

    # Stratified CV
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    last_model = None
    last_X_test = None
    last_y_test = None

    plt.figure(figsize=(8, 6))

    # Cross-validation loop to compute ROC and AUC
    for train_idx, test_idx in cv.split(X_scaled, y):
        # Split the data
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize the OLS model (Linear Regression for binary classification)
        model = LinearRegression()

        # Fit the model
        model.fit(X_train, y_train)

        last_model = model
        last_X_test = X_test
        last_y_test = y_test

        # Predict continuous scores and clip to [0, 1]
        y_prob_ols = model.predict(X_test)
        y_prob_ols = np.clip(y_prob_ols, 0, 1)

        # Calculate ROC curve
        fpr_ols, tpr_ols, _ = roc_curve(y_test, y_prob_ols)
        roc_auc_ols = auc(fpr_ols, tpr_ols)
        auc_scores_ols.append(roc_auc_ols)

        # Interpolate to the same number of points for mean ROC calculation
        mean_tpr_ols = np.interp(mean_fpr, fpr_ols, tpr_ols)
        mean_tpr_ols[0] = 0.0  # Ensure the first point is (0,0)

        # Store TPR curve
        all_tpr_ols.append(mean_tpr_ols)

        # Plot ROC curve for the current fold
        plt.plot(
            fpr_ols,
            tpr_ols,
            lw=1.5,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc_ols:.3f})",
        )
        fold += 1

    # Calculate the mean ROC curve and AUC
    mean_tpr_ols = np.mean(all_tpr_ols, axis=0)
    mean_auc_ols = auc(mean_fpr, mean_tpr_ols)

    # Calculate mean and 95% confidence interval for AUC
    mean_auc_ols = np.mean(auc_scores_ols)
    std_auc_ols = np.std(auc_scores_ols, ddof=1)
    n_folds = len(auc_scores_ols)
    se_auc_ols = std_auc_ols / np.sqrt(n_folds)
    ci_low_ols, ci_high_ols = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc_ols, scale=se_auc_ols
    )

    # Calculate standard deviation for the ROC curve
    std_tpr_ols = np.std(all_tpr_ols, axis=0)
    tpr_upper_ols = np.minimum(mean_tpr_ols + std_tpr_ols, 1)
    tpr_lower_ols = np.maximum(mean_tpr_ols - std_tpr_ols, 0)

    # Plot the mean ROC curve with confidence interval band
    plt.plot(
        mean_fpr,
        mean_tpr_ols,
        color="b",
        linestyle="-",
        lw=2,
        label=(
            f"Mean ROC (AUC = {mean_auc_ols:.3f}, "
            f"95% CI = [{ci_low_ols:.3f}, {ci_high_ols:.3f}])"
        ),
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower_ols,
        tpr_upper_ols,
        color="blue",
        alpha=0.2,
        label="±1 std. dev.",
    )

    # Plot the diagonal line (random classifier) for reference
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    # Set the plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "Receiver Operating Characteristic (ROC) for Ordinary Least Squares "
        "(with Stratified Cross-Validation)"
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save ROC figure
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Evaluate on the last fold's test split
    if last_model is not None:
        evaluate_on_test(last_model, last_X_test, last_y_test, output_prefix)

    plt.show()


def evaluate_on_test(model, X_test, y_test, output_prefix: str = "ols"):
    """
    Compute confusion matrix, accuracy, sensitivity, specificity,
    and 95% CIs; save results as CSVs.
    """
    # Continuous scores → threshold at 0.5
    y_prob_ols = model.predict(X_test)
    y_prob_ols = np.clip(y_prob_ols, 0, 1)
    threshold = 0.5
    y_pred_ols = (y_prob_ols >= threshold).astype(int)

    # Compute confusion matrix
    cm_ols = confusion_matrix(y_test, y_pred_ols)
    print("Confusion Matrix:")
    print(cm_ols)

    tn_ols, fp_ols, fn_ols, tp_ols = cm_ols.ravel()

    # Compute accuracy
    acc_ols = accuracy_score(y_test, y_pred_ols)
    print(f"Accuracy: {acc_ols:.3f}")

    n = len(y_test)
    se = np.sqrt((acc_ols * (1 - acc_ols)) / n)
    ci_low_ols = acc_ols - 1.96 * se
    ci_high_ols = acc_ols + 1.96 * se
    print(f"95% Confidence interval for accuracy: [{ci_low_ols:.3f}, {ci_high_ols:.3f}]")

    # Sensitivity and Specificity
    sensitivity_ols = tp_ols / (tp_ols + fn_ols)
    specificity_ols = tn_ols / (tn_ols + fp_ols)
    print(f"Sensitivity: {sensitivity_ols:.3f}")
    print(f"Specificity: {specificity_ols:.3f}")

    # 95% CI for Sensitivity and Specificity (Wilson score interval)
    sens_low_ols, sens_upp_ols = proportion_confint(
        tp_ols, tp_ols + fn_ols, alpha=0.05, method="wilson"
    )
    spec_low_ols, spec_upp_ols = proportion_confint(
        tn_ols, tn_ols + fp_ols, alpha=0.05, method="wilson"
    )
    print(f"Sensitivity 95% CI: [{sens_low_ols:.3f}, {sens_upp_ols:.3f}]")
    print(f"Specificity 95% CI: [{spec_low_ols:.3f}, {spec_upp_ols:.3f}]")

    # Perform a binomial test to get p-value (test if accuracy > 0.5)
    p_value = binomtest((y_pred_ols == y_test).sum(), n, 0.5, alternative="greater")
    print(f"P-value for accuracy > 0.5: {p_value.pvalue:.3f}")

    # Save to CSVs
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(
        cm_ols,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"],
    ).to_csv(cm_path)

    results_ols = pd.DataFrame(
        [
            {
                "model": output_prefix,
                "accuracy": acc_ols,
                "acc_ci_lower": ci_low_ols,
                "acc_ci_upper": ci_high_ols,
                "n_samples": n,
                "sensitivity": sensitivity_ols,
                "sens_ci_lower": sens_low_ols,
                "sens_ci_upper": sens_upp_ols,
                "specificity": specificity_ols,
                "spec_ci_lower": spec_low_ols,
                "spec_ci_upper": spec_upp_ols,
                "p_value_acc_gt_0.5": p_value.pvalue,
            }
        ]
    )
    metrics_path = tables_dir / f"{output_prefix}_metrics.csv"
    results_ols.to_csv(metrics_path, index=False)

    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")


def main():
    """
    Convenience entry point so you can run:

        python -m src.arvc_ml.ols
    """
    run_ols_cv()


if __name__ == "__main__":
    main()

