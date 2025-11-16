"""
Hyperparameter-tuned Decision Tree multimodal model for ARVC.

- 5-fold Stratified CV with inner 3-fold GridSearchCV
- ROC curves per fold + mean ROC with 95% CI
- Confusion matrix, accuracy, sensitivity, specificity + 95% CIs

Data loading and scaling is handled by load_knn_imputed_multimodal() in data.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
)

from .data import load_knn_imputed_multimodal
from .config import REPORTS_DIR, RANDOM_STATE


# Parameter grid from your notebook
PARAM_GRID = {
    "max_depth": [None, 3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"],
}


def run_decision_tree_cv(
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    output_prefix: str = "decision_tree",
):
    """
    Run stratified K-fold CV with Decision Tree + GridSearchCV and plot ROC curves.
    """

    # 🔁 Replace CSV + manual scaling with shared loader
    X_scaled, y, feature_names = load_knn_imputed_multimodal()

    all_tpr_dt = []
    auc_scores_dt = []
    mean_fpr = np.linspace(0, 1, 100)
    best_params_list = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    last_best_model = None
    last_X_test = None
    last_y_test = None

    plt.figure(figsize=(8, 6))

    # Outer CV: each fold gets its own GridSearchCV on the training set
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        base_tree = DecisionTreeClassifier(random_state=random_state)

        grid_search = GridSearchCV(
            estimator=base_tree,
            param_grid=PARAM_GRID,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        best_dt = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_params_list.append(best_params)

        print(f"Fold {fold} - Best parameters: {best_params}")

        last_best_model = best_dt
        last_X_test = X_test
        last_y_test = y_test

        # ROC on this fold
        y_prob_dt = best_dt.predict_proba(X_test)[:, 1]
        fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
        roc_auc_dt = auc(fpr_dt, tpr_dt)
        auc_scores_dt.append(roc_auc_dt)

        mean_tpr_dt = np.interp(mean_fpr, fpr_dt, tpr_dt)
        mean_tpr_dt[0] = 0.0
        all_tpr_dt.append(mean_tpr_dt)

        plt.plot(
            fpr_dt,
            tpr_dt,
            lw=1.5,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc_dt:.3f})",
        )

        fold += 1

    # AUC summary across folds
    mean_auc_dt = np.mean(auc_scores_dt)
    std_auc_dt = np.std(auc_scores_dt, ddof=1)
    n_folds = len(auc_scores_dt)
    se_auc_dt = std_auc_dt / np.sqrt(n_folds)
    ci_low_dt, ci_high_dt = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc_dt, scale=se_auc_dt
    )

    # Mean ROC curve
    mean_tpr_dt = np.mean(all_tpr_dt, axis=0)
    mean_auc_dt_curve = auc(mean_fpr, mean_tpr_dt)

    std_tpr_dt = np.std(all_tpr_dt, axis=0)
    tpr_upper_dt = np.minimum(mean_tpr_dt + std_tpr_dt, 1)
    tpr_lower_dt = np.maximum(mean_tpr_dt - std_tpr_dt, 0)

    plt.plot(
        mean_fpr,
        mean_tpr_dt,
        color="b",
        linestyle="-",
        lw=2,
        label=(
            f"Mean ROC (AUC = {mean_auc_dt_curve:.3f}, "
            f"95% CI = [{ci_low_dt:.3f}, {ci_high_dt:.3f}])"
        ),
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower_dt,
        tpr_upper_dt,
        color="blue",
        alpha=0.2,
        label="±1 std. dev.",
    )

    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for Decision Tree with Hyperparameter Tuning (Cross-Validation)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save ROC figure
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Summary of best params across folds
    print("\nSummary of best parameters across all folds:")
    for i, params in enumerate(best_params_list, 1):
        print(f"Fold {i}: {params}")

    print("\nMost common best parameters across folds:")
    for param in PARAM_GRID.keys():
        vals = [fold_params[param] for fold_params in best_params_list]
        counts = Counter(vals)
        most_common = counts.most_common(1)[0]
        print(
            f"{param}: most common value = {most_common[0]} "
            f"(appeared in {most_common[1]}/{len(best_params_list)} folds)"
        )

    # Evaluate on the last fold’s test split (mirrors your notebook pattern)
    if last_best_model is not None:
        evaluate_on_test(
            last_best_model, last_X_test, last_y_test, output_prefix=output_prefix
        )

    plt.show()


def evaluate_on_test(
    model, X_test, y_test, output_prefix: str = "decision_tree"
):
    """
    Compute confusion matrix, accuracy, sensitivity, specificity,
    and 95% CIs; save results as CSVs.
    """

    y_prob_dt = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred_dt = (y_prob_dt >= threshold).astype(int)

    # Confusion matrix
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    print("Confusion Matrix:")
    print(cm_dt)

    tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()

    # Accuracy
    acc_dt = accuracy_score(y_test, y_pred_dt)
    print(f"Accuracy: {acc_dt:.3f}")

    n = len(y_test)
    se = np.sqrt((acc_dt * (1 - acc_dt)) / n)
    ci_low_dt = acc_dt - 1.96 * se
    ci_high_dt = acc_dt + 1.96 * se
    print(f"95% Confidence interval for accuracy: [{ci_low_dt:.3f}, {ci_high_dt:.3f}]")

    # Sensitivity & Specificity
    sensitivity_dt = tp_dt / (tp_dt + fn_dt)
    specificity_dt = tn_dt / (tn_dt + fp_dt)
    print(f"Sensitivity: {sensitivity_dt:.3f}")
    print(f"Specificity: {specificity_dt:.3f}")

    # 95% CI via Wilson interval
    sens_low_dt, sens_upp_dt = proportion_confint(
        tp_dt, tp_dt + fn_dt, alpha=0.05, method="wilson"
    )
    spec_low_dt, spec_upp_dt = proportion_confint(
        tn_dt, tn_dt + fp_dt, alpha=0.05, method="wilson"
    )
    print(f"Sensitivity 95% CI: [{sens_low_dt:.3f}, {sens_upp_dt:.3f}]")
    print(f"Specificity 95% CI: [{spec_low_dt:.3f}, {spec_upp_dt:.3f}]")

    # Binomial test: is accuracy > 0.5?
    p_value = binomtest((y_pred_dt == y_test).sum(), n, 0.5, alternative="greater")
    print(f"P-value for accuracy > 0.5: {p_value.pvalue:.3f}")

    # Save outputs
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(
        cm_dt,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"],
    ).to_csv(cm_path)

    results_dt = pd.DataFrame(
        [
            {
                "model": output_prefix,
                "accuracy": acc_dt,
                "acc_ci_lower": ci_low_dt,
                "acc_ci_upper": ci_high_dt,
                "n_samples": n,
                "sensitivity": sensitivity_dt,
                "sens_ci_lower": sens_low_dt,
                "sens_ci_upper": sens_upp_dt,
                "specificity": specificity_dt,
                "spec_ci_lower": spec_low_dt,
                "spec_ci_upper": spec_upp_dt,
                "p_value_acc_gt_0.5": p_value.pvalue,
            }
        ]
    )
    metrics_path = tables_dir / f"{output_prefix}_metrics.csv"
    results_dt.to_csv(metrics_path, index=False)

    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")


def main():
    """
    Convenience entry point so you can run:

        python -m src.arvc_ml.decision_tree
    """
    run_decision_tree_cv()


if __name__ == "__main__":
    main()

