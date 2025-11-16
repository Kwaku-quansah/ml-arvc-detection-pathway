"""
Hyperparameter-tuned Lasso Logistic Regression multimodal model for ARVC.

- 5-fold Stratified CV with inner 3-fold GridSearchCV
- L1-penalized LogisticRegression (Lasso) over a grid of C, class_weight, etc.
- ROC curves per fold + mean ROC with 95% CI
- Confusion matrix, accuracy, sensitivity, specificity + 95% CIs
- Coefficient-based feature importance across folds

Data loading and scaling is handled by load_knn_imputed_multimodal() in data.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    accuracy_score,
    make_scorer,
)

from .data import load_knn_imputed_multimodal
from .config import REPORTS_DIR, RANDOM_STATE


# Parameter grid from your notebook
PARAM_GRID = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100],  # inverse of regularization strength
    "class_weight": [None, "balanced"],
    "fit_intercept": [True, False],
    "tol": [1e-4, 1e-5],
    "solver": ["liblinear"],  # compatible with L1
    "penalty": ["l1"],  # L1 penalty (Lasso)
    "max_iter": [1000, 2000],
}


def run_lasso_logistic_cv(
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    output_prefix: str = "lasso_logistic",
):
    """
    Run stratified K-fold CV with L1-penalized Logistic Regression (Lasso) and plot ROC curves.
    """

    # 🔁 Replaces df = pd.read_csv(...), manual split, scaling, etc.
    X_scaled, y, feature_names = load_knn_imputed_multimodal()

    all_tpr_ls = []
    auc_scores_ls = []
    mean_fpr = np.linspace(0, 1, 100)
    best_params_list = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    last_best_model = None
    last_X_test = None
    last_y_test = None

    plt.figure(figsize=(8, 6))

    # Perform Stratified Cross-Validation with hyperparameter tuning
    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Define the scoring metric (ROC AUC)
        roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)

        # GridSearchCV over L1 LogisticRegression
        grid_search = GridSearchCV(
            estimator=LogisticRegression(random_state=random_state),
            param_grid=PARAM_GRID,
            scoring=roc_auc_scorer,
            cv=3,
            n_jobs=-1,
            verbose=1,
        )

        print(f"Performing grid search for fold {fold}...")
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        print(f"Fold {fold} Best Parameters: {best_params}")
        print(f"Fold {fold} Best CV Score: {grid_search.best_score_:.3f}")

        best_lasso = grid_search.best_estimator_

        last_best_model = best_lasso
        last_X_test = X_test
        last_y_test = y_test

        # Predictions for ROC
        y_prob_ls = best_lasso.predict_proba(X_test)[:, 1]

        fpr_ls, tpr_ls, _ = roc_curve(y_test, y_prob_ls)
        roc_auc_ls = auc(fpr_ls, tpr_ls)
        auc_scores_ls.append(roc_auc_ls)

        mean_tpr_ls = np.interp(mean_fpr, fpr_ls, tpr_ls)
        mean_tpr_ls[0] = 0.0
        all_tpr_ls.append(mean_tpr_ls)

        plt.plot(
            fpr_ls,
            tpr_ls,
            lw=1.5,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc_ls:.3f})",
        )

        fold += 1

    # AUC summary across folds
    mean_auc_ls = np.mean(auc_scores_ls)
    std_auc_ls = np.std(auc_scores_ls, ddof=1)
    n_folds = len(auc_scores_ls)
    se_auc_ls = std_auc_ls / np.sqrt(n_folds)
    ci_low_ls, ci_high_ls = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc_ls, scale=se_auc_ls
    )

    # Mean ROC curve
    mean_tpr_ls = np.mean(all_tpr_ls, axis=0)
    mean_auc_ls_curve = auc(mean_fpr, mean_tpr_ls)

    std_tpr_ls = np.std(all_tpr_ls, axis=0)
    tpr_upper_ls = np.minimum(mean_tpr_ls + std_tpr_ls, 1)
    tpr_lower_ls = np.maximum(mean_tpr_ls - std_tpr_ls, 0)

    # Mean ROC + CI band
    plt.plot(
        mean_fpr,
        mean_tpr_ls,
        color="b",
        linestyle="-",
        lw=2,
        label=(
            f"Mean ROC (AUC = {mean_auc_ls_curve:.3f}, "
            f"95% CI= [{ci_low_ls:.3f}, {ci_high_ls:.3f}])"
        ),
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower_ls,
        tpr_upper_ls,
        color="blue",
        alpha=0.2,
        label="±1 std. dev.",
    )

    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for Tuned Lasso Logistic Regression with Cross-Validation")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save ROC figure
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Feature importance analysis (coefficients across folds)
    analyze_feature_importance(
        X_scaled=X_scaled,
        y=y,
        feature_names=feature_names,
        best_params_list=best_params_list,
        n_splits=n_splits,
        random_state=random_state,
    )

    # Summary of best params across folds
    print("\nSummary of best parameters across all folds:")
    for param in PARAM_GRID.keys():
        param_values = [fold_params[param] for fold_params in best_params_list]
        most_common = Counter(param_values).most_common(1)[0]
        print(
            f"{param}: most common value = {most_common[0]} "
            f"(appeared in {most_common[1]}/{len(best_params_list)} folds)"
        )

    final_best_params = {
        param: Counter([fold_params[param] for fold_params in best_params_list])
        .most_common(1)[0][0]
        for param in PARAM_GRID.keys()
    }
    print(f"\nFinal best parameters: {final_best_params}")

    # Evaluate on the last fold's test split
    if last_best_model is not None:
        evaluate_on_test(
            last_best_model, last_X_test, last_y_test, output_prefix=output_prefix
        )

    plt.show()


def analyze_feature_importance(
    X_scaled,
    y,
    feature_names,
    best_params_list,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
):
    """
    Refit L1 LogisticRegression on each fold's training data using its best params,
    aggregate coefficients, and report top features.
    """

    print("\nFeature Importance Analysis:")

    n_folds = len(best_params_list)
    n_features = X_scaled.shape[1]
    coef_matrix = np.zeros((n_folds, n_features))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for i, (train_idx, _) in enumerate(cv.split(X_scaled, y)):
        if i >= n_folds:
            break  # Just in case, though they should match

        X_train = X_scaled[train_idx]
        y_train = y[train_idx]

        best_params = best_params_list[i]

        best_model = LogisticRegression(
            **best_params,
            random_state=random_state,
        )
        best_model.fit(X_train, y_train)

        coef_matrix[i, :] = best_model.coef_[0]

    mean_coefs = np.mean(coef_matrix, axis=0)
    std_coefs = np.std(coef_matrix, axis=0)

    # Sort by absolute coefficient
    sorted_idx = np.argsort(np.abs(mean_coefs))[::-1]
    top_n = min(10, len(sorted_idx))

    print("\nTop important features (based on absolute coefficient value):")
    for i in range(top_n):
        idx = sorted_idx[i]
        print(f"{feature_names[idx]}: {mean_coefs[idx]:.3f} ± {std_coefs[idx]:.3f}")

    non_zero_counts = np.sum(coef_matrix != 0, axis=0)
    selected_features = np.where(non_zero_counts > n_folds / 2)[0]
    print(
        f"\nFeatures selected in more than half of the folds: "
        f"{len(selected_features)} out of {n_features}"
    )


def evaluate_on_test(
    model, X_test, y_test, output_prefix: str = "lasso_logistic"
):
    """
    Compute confusion matrix, accuracy, sensitivity, specificity,
    and 95% CIs; save results as CSVs.
    """

    y_prob_ls = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred_ls = (y_prob_ls >= threshold).astype(int)

    cm_ls = confusion_matrix(y_test, y_pred_ls)
    print("Confusion Matrix:")
    print(cm_ls)

    tn_ls, fp_ls, fn_ls, tp_ls = cm_ls.ravel()

    acc_ls = accuracy_score(y_test, y_pred_ls)
    print(f"Accuracy: {acc_ls:.3f}")

    n = len(y_test)
    se = np.sqrt((acc_ls * (1 - acc_ls)) / n)
    ci_low_ls = acc_ls - 1.96 * se
    ci_high_ls = acc_ls + 1.96 * se
    print(f"95% Confidence interval for accuracy: [{ci_low_ls:.3f}, {ci_high_ls:.3f}]")

    sensitivity_ls = tp_ls / (tp_ls + fn_ls)
    specificity_ls = tn_ls / (tn_ls + fp_ls)
    print(f"Sensitivity: {sensitivity_ls:.3f}")
    print(f"Specificity: {specificity_ls:.3f}")

    sens_low_ls, sens_upp_ls = proportion_confint(
        tp_ls, tp_ls + fn_ls, alpha=0.05, method="wilson"
    )
    spec_low_ls, spec_upp_ls = proportion_confint(
        tn_ls, tn_ls + fp_ls, alpha=0.05, method="wilson"
    )
    print(f"Sensitivity 95% CI: [{sens_low_ls:.3f}, {sens_upp_ls:.3f}]")
    print(f"Specificity 95% CI: [{spec_low_ls:.3f}, {spec_upp_ls:.3f}]")

    p_value = binomtest((y_pred_ls == y_test).sum(), n, 0.5, alternative="greater")
    print(f"P-value for accuracy > 0.5: {p_value.pvalue:.3f}")

    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(
        cm_ls,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"],
    ).to_csv(cm_path)

    results_ls = pd.DataFrame(
        [
            {
                "model": output_prefix,
                "accuracy": acc_ls,
                "acc_ci_lower": ci_low_ls,
                "acc_ci_upper": ci_high_ls,
                "n_samples": n,
                "sensitivity": sensitivity_ls,
                "sens_ci_lower": sens_low_ls,
                "sens_ci_upper": sens_upp_ls,
                "specificity": specificity_ls,
                "spec_ci_lower": spec_low_ls,
                "spec_ci_upper": spec_upp_ls,
                "p_value_acc_gt_0.5": p_value.pvalue,
            }
        ]
    )
    metrics_path = tables_dir / f"{output_prefix}_metrics.csv"
    results_ls.to_csv(metrics_path, index=False)

    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")


def main():
    """
    Convenience entry point so you can run:

        python -m src.arvc_ml.lasso_logistic
    """
    run_lasso_logistic_cv()


if __name__ == "__main__":
    main()

