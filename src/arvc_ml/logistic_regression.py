"""
Logistic Regression multimodal model for ARVC.

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

from sklearn.linear_model import LogisticRegression
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


def run_logistic_regression_cv(
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    output_prefix: str = "logistic_regression",
):
    """
    Run stratified K-fold CV with Logistic Regression and plot ROC curves.
    """

    # 🔁 Replaces df = pd.read_csv(...), data/target, X, y, scaler, etc.
    X_scaled, y, feature_names = load_knn_imputed_multimodal()

    # Containers for ROC and AUC across folds
    all_tpr_lr = []
    auc_scores_lr = []
    mean_fpr = np.linspace(0, 1, 100)

    # Stratified CV
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    last_model = None
    last_X_test = None
    last_y_test = None

    plt.figure(figsize=(8, 6))

    # Perform Stratified Cross-Validation and plot ROC curve for each fold
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Initialize and train the Logistic Regression model
        model = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            penalty="l2",
            solver="lbfgs",
        )
        model.fit(X_train, y_train)

        last_model = model
        last_X_test = X_test
        last_y_test = y_test

        # Predicted probabilities for positive class
        y_prob_lr = model.predict_proba(X_test)[:, 1]

        # ROC for current fold
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
        roc_auc_lr = auc(fpr_lr, tpr_lr)
        auc_scores_lr.append(roc_auc_lr)

        # Interpolate TPR onto mean_fpr grid
        mean_tpr_lr = np.interp(mean_fpr, fpr_lr, tpr_lr)
        mean_tpr_lr[0] = 0.0
        all_tpr_lr.append(mean_tpr_lr)

        # Plot this fold's ROC
        plt.plot(
            fpr_lr,
            tpr_lr,
            lw=1.5,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc_lr:.3f})",
        )
        fold += 1

    # Mean ROC and AUC across folds
    mean_tpr_lr = np.mean(all_tpr_lr, axis=0)
    mean_auc_lr = auc(mean_fpr, mean_tpr_lr)

    # 95% CI for AUC across folds
    mean_auc_lr = np.mean(auc_scores_lr)
    std_auc_lr = np.std(auc_scores_lr, ddof=1)
    n_folds = len(auc_scores_lr)
    se_auc_lr = std_auc_lr / np.sqrt(n_folds)
    ci_low_lr, ci_high_lr = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc_lr, scale=se_auc_lr
    )

    # Standard deviation band for TPR at each FPR
    std_tpr_lr = np.std(all_tpr_lr, axis=0)
    tpr_upper_lr = np.minimum(mean_tpr_lr + std_tpr_lr, 1)
    tpr_lower_lr = np.maximum(mean_tpr_lr - std_tpr_lr, 0)

    # Mean ROC curve with CI band
    plt.plot(
        mean_fpr,
        mean_tpr_lr,
        color="b",
        linestyle="-",
        lw=2,
        label=(
            f"Mean ROC (AUC = {mean_auc_lr:.3f}, "
            f"95% CI = [{ci_low_lr:.3f}, {ci_high_lr:.3f}])"
        ),
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower_lr,
        tpr_upper_lr,
        color="blue",
        alpha=0.2,
        label="±1 std. dev.",
    )

    # Diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    # Plot formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "Receiver Operating Characteristic (ROC) for Logistic Regression "
        "(with Stratified Cross-Validation)"
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save ROC figure under reports/figures/
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Evaluate on the last fold's test split
    if last_model is not None:
        evaluate_on_test(last_model, last_X_test, last_y_test, output_prefix)

    plt.show()


def evaluate_on_test(
    model, X_test, y_test, output_prefix: str = "logistic_regression"
):
    """
    Compute confusion matrix, accuracy, sensitivity, specificity,
    and 95% CIs; save results as CSVs.
    """
    y_prob_lr = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred_lr = (y_prob_lr >= threshold).astype(int)

    # Confusion matrix
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    print("Confusion Matrix:")
    print(cm_lr)

    tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()

    # Accuracy
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Accuracy: {acc_lr:.3f}")

    n = len(y_test)
    se = np.sqrt((acc_lr * (1 - acc_lr)) / n)
    ci_low_lr = acc_lr - 1.96 * se
    ci_high_lr = acc_lr + 1.96 * se
    print(f"95% Confidence interval for accuracy: [{ci_low_lr:.3f}, {ci_high_lr:.3f}]")

    # Sensitivity & Specificity
    sensitivity_lr = tp_lr / (tp_lr + fn_lr)
    specificity_lr = tn_lr / (tn_lr + fp_lr)
    print(f"Sensitivity: {sensitivity_lr:.3f}")
    print(f"Specificity: {specificity_lr:.3f}")

    # 95% CI via Wilson interval
    sens_low_lr, sens_upp_lr = proportion_confint(
        tp_lr, tp_lr + fn_lr, alpha=0.05, method="wilson"
    )
    spec_low_lr, spec_upp_lr = proportion_confint(
        tn_lr, tn_lr + fp_lr, alpha=0.05, method="wilson"
    )
    print(f"Sensitivity 95% CI: [{sens_low_lr:.3f}, {sens_upp_lr:.3f}]")
    print(f"Specificity 95% CI: [{spec_low_lr:.3f}, {spec_upp_lr:.3f}]")

    # Binomial test: is accuracy > 0.5?
    p_value = binomtest((y_pred_lr == y_test).sum(), n, 0.5, alternative="greater")
    print(f"P-value for accuracy > 0.5: {p_value.pvalue:.3f}")

    # Save outputs
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(
        cm_lr,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"],
    ).to_csv(cm_path)

    results_lr = pd.DataFrame(
        [
            {
                "model": output_prefix,
                "accuracy": acc_lr,
                "acc_ci_lower": ci_low_lr,
                "acc_ci_upper": ci_high_lr,
                "n_samples": n,
                "sensitivity": sensitivity_lr,
                "sens_ci_lower": sens_low_lr,
                "sens_ci_upper": sens_upp_lr,
                "specificity": specificity_lr,
                "spec_ci_lower": spec_low_lr,
                "spec_ci_upper": spec_upp_lr,
                "p_value_acc_gt_0.5": p_value.pvalue,
            }
        ]
    )
    metrics_path = tables_dir / f"{output_prefix}_metrics.csv"
    results_lr.to_csv(metrics_path, index=False)

    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")


def main():
    """
    Convenience entry point so you can run:

        python -m src.arvc_ml.logistic_regression
    """
    run_logistic_regression_cv()


if __name__ == "__main__":
    main()

