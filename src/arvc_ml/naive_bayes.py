"""
Naive Bayes multimodal model for ARVC.

- 5-fold Stratified CV using GaussianNB
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

from sklearn.naive_bayes import GaussianNB
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


def run_naive_bayes_cv(
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    output_prefix: str = "naive_bayes",
):
    """
    Run stratified K-fold CV with Gaussian Naive Bayes and plot ROC curves.
    """

    # 🔁 Replaces df = pd.read_csv(...), data/target, X, y, scaler, etc.
    X_scaled, y, feature_names = load_knn_imputed_multimodal()

    # Containers for ROC and AUC across folds
    all_tpr_nb = []
    auc_scores_nb = []
    mean_fpr = np.linspace(0, 1, 100)

    # Stratified CV
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize model
    nb_model = GaussianNB()

    fold = 1
    last_model = None
    last_X_test = None
    last_y_test = None

    plt.figure(figsize=(8, 6))

    # Perform Stratified Cross-Validation and plot ROC curve for each fold
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train the Naive Bayes model
        nb_model.fit(X_train, y_train)

        last_model = nb_model
        last_X_test = X_test
        last_y_test = y_test

        # Predicted probabilities for positive class
        y_prob_nb = nb_model.predict_proba(X_test)[:, 1]

        # ROC for current fold
        fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
        roc_auc_nb = auc(fpr_nb, tpr_nb)
        auc_scores_nb.append(roc_auc_nb)

        # Interpolate TPR onto mean_fpr grid
        mean_tpr_nb = np.interp(mean_fpr, fpr_nb, tpr_nb)
        mean_tpr_nb[0] = 0.0
        all_tpr_nb.append(mean_tpr_nb)

        # Plot this fold's ROC
        plt.plot(
            fpr_nb,
            tpr_nb,
            lw=1.5,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc_nb:.3f})",
        )
        fold += 1

    # Mean ROC and AUC across folds
    mean_tpr_nb = np.mean(all_tpr_nb, axis=0)
    mean_auc_nb = auc(mean_fpr, mean_tpr_nb)

    # 95% CI for AUC across folds
    mean_auc_nb = np.mean(auc_scores_nb)
    std_auc_nb = np.std(auc_scores_nb, ddof=1)
    n_folds = len(auc_scores_nb)
    se_auc_nb = std_auc_nb / np.sqrt(n_folds)
    ci_low_nb, ci_high_nb = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc_nb, scale=se_auc_nb
    )

    # Standard deviation band for TPR at each FPR
    std_tpr_nb = np.std(all_tpr_nb, axis=0)
    tpr_upper_nb = np.minimum(mean_tpr_nb + std_tpr_nb, 1)
    tpr_lower_nb = np.maximum(mean_tpr_nb - std_tpr_nb, 0)

    # Mean ROC curve
    plt.plot(
        mean_fpr,
        mean_tpr_nb,
        color="b",
        linestyle="-",
        lw=2,
        label=(
            f"Mean ROC (AUC = {mean_auc_nb:.3f}, "
            f"95% CI = [{ci_low_nb:.3f}, {ci_high_nb:.3f}])"
        ),
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower_nb,
        tpr_upper_nb,
        color="blue",
        alpha=0.2,
        label="±1 std. dev.",
    )

    # Diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        "Receiver Operating Characteristic (ROC) for Gaussian Naive Bayes "
        "(with Stratified Cross-Validation)"
    )
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Evaluate on the last fold's test split (mirroring your notebook pattern)
    if last_model is not None:
        evaluate_on_test(last_model, last_X_test, last_y_test, output_prefix)

    plt.show()


def evaluate_on_test(model, X_test, y_test, output_prefix: str = "naive_bayes"):
    """
    Compute confusion matrix, accuracy, sensitivity, specificity,
    and 95% CIs; save results as CSVs.
    """
    # Predicted probabilities and class labels
    y_prob_nb = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred_nb = (y_prob_nb >= threshold).astype(int)

    # Confusion matrix
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    print("Confusion Matrix:")
    print(cm_nb)

    tn_nb, fp_nb, fn_nb, tp_nb = cm_nb.ravel()

    # Accuracy
    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Accuracy: {acc_nb:.3f}")

    n = len(y_test)
    se = np.sqrt((acc_nb * (1 - acc_nb)) / n)
    ci_low_nb = acc_nb - 1.96 * se
    ci_high_nb = acc_nb + 1.96 * se
    print(f"95% Confidence interval for accuracy: [{ci_low_nb:.3f}, {ci_high_nb:.3f}]")

    # Sensitivity & Specificity
    sensitivity_nb = tp_nb / (tp_nb + fn_nb)
    specificity_nb = tn_nb / (tn_nb + fp_nb)
    print(f"Sensitivity: {sensitivity_nb:.3f}")
    print(f"Specificity: {specificity_nb:.3f}")

    # 95% CI via Wilson interval
    sens_low_nb, sens_upp_nb = proportion_confint(
        tp_nb, tp_nb + fn_nb, alpha=0.05, method="wilson"
    )
    spec_low_nb, spec_upp_nb = proportion_confint(
        tn_nb, tn_nb + fp_nb, alpha=0.05, method="wilson"
    )
    print(f"Sensitivity 95% CI: [{sens_low_nb:.3f}, {sens_upp_nb:.3f}]")
    print(f"Specificity 95% CI: [{spec_low_nb:.3f}, {spec_upp_nb:.3f}]")

    # Binomial test: is accuracy > 0.5?
    p_value = binomtest((y_pred_nb == y_test).sum(), n, 0.5, alternative="greater")
    print(f"P-value for accuracy > 0.5: {p_value.pvalue:.3f}")

    # Save outputs
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(
        cm_nb,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"],
    ).to_csv(cm_path)

    results_nb = pd.DataFrame(
        [
            {
                "model": output_prefix,
                "accuracy": acc_nb,
                "acc_ci_lower": ci_low_nb,
                "acc_ci_upper": ci_high_nb,
                "n_samples": n,
                "sensitivity": sensitivity_nb,
                "sens_ci_lower": sens_low_nb,
                "sens_ci_upper": sens_upp_nb,
                "specificity": specificity_nb,
                "spec_ci_lower": spec_low_nb,
                "spec_ci_upper": spec_upp_nb,
                "p_value_acc_gt_0.5": p_value.pvalue,
            }
        ]
    )
    metrics_path = tables_dir / f"{output_prefix}_metrics.csv"
    results_nb.to_csv(metrics_path, index=False)

    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")


def main():
    """
    Convenience entry point so you can run:

        python -m src.arvc_ml.naive_bayes
    """
    run_naive_bayes_cv()


if __name__ == "__main__":
    main()

