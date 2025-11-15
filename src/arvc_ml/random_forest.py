"""
Random Forest multimodal model with hyperparameter tuning, cross-validated ROC,
and test-set performance metrics for ARVC diagnosis.

- Loads a KNN-imputed multimodal dataset from PROCESSED_DIR
- Runs 5-fold Stratified CV with GridSearch hyperparameter tuning
- Plots fold-wise ROC curves + mean ROC with 95% CI
- Computes confusion matrix, accuracy, sensitivity, specificity + 95% CIs
- Writes figures and tables into the reports/ directory
"""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    accuracy_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED_DIR, REPORTS_DIR


# Hyperparameter grid for Random Forest
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}


def load_multimodal_dataset(csv_path: Path):
    """
    Load the KNN-imputed multimodal dataset.

    Expects a column named 'ARVC diagnosed' as the target.
    """
    df = pd.read_csv(csv_path)

    data = df.drop(columns=["ARVC diagnosed"])
    target = df["ARVC diagnosed"]
    feature_names = list(data.columns)

    X = data.to_numpy()
    y = target.to_numpy()

    return X, y, feature_names


def run_random_forest_cv(
    csv_path: Path,
    output_prefix: str = "random_forest",
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Run stratified K-fold cross-validation with hyperparameter tuning,
    plot ROC curves, and compute performance metrics.
    """

    # Load data
    X, y, feature_names = load_multimodal_dataset(csv_path)
    print("Features:")
    print(feature_names)
    print("\nTarget (first 5):")
    print(y[:5])

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Containers for ROC and AUC across folds
    all_tpr_rf = []
    auc_scores_rf = []
    mean_fpr = np.linspace(0, 1, 100)
    best_params_list = []

    # Stratified KFold setup
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    best_rf_last_fold = None
    X_test_last = None
    y_test_last = None

    plt.figure(figsize=(8, 6))

    for train_idx, test_idx in cv.split(X_scaled, y):
        # Split for current fold
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ROC AUC scorer
        roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)

        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=random_state),
            param_grid=PARAM_GRID,
            scoring=roc_auc_scorer,
            cv=3,
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        print(f"Fold {fold} Best Parameters: {best_params}")

        best_rf = grid_search.best_estimator_
        best_rf_last_fold = best_rf
        X_test_last = X_test
        y_test_last = y_test

        # Probabilities and ROC for the current fold
        y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        auc_scores_rf.append(roc_auc_rf)

        # Interpolate TPR to a common FPR grid
        mean_tpr_rf = np.interp(mean_fpr, fpr_rf, tpr_rf)
        mean_tpr_rf[0] = 0.0

        all_tpr_rf.append(mean_tpr_rf)

        # Plot ROC for this fold
        plt.plot(
            fpr_rf,
            tpr_rf,
            lw=1,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc_rf:.3f})",
        )
        fold += 1

    # Mean ROC and AUC across folds
    mean_tpr_rf = np.mean(all_tpr_rf, axis=0)
    mean_auc_rf = np.mean(auc_scores_rf)

    # 95% CI for AUC across folds (t-based)
    std_auc_rf = np.std(auc_scores_rf, ddof=1)
    n_folds = len(auc_scores_rf)
    se_auc_rf = std_auc_rf / np.sqrt(n_folds)
    ci_low_rf, ci_high_rf = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc_rf, scale=se_auc_rf
    )

    # Standard deviation for TPR curve
    std_tpr_rf = np.std(all_tpr_rf, axis=0)
    tpr_upper_rf = np.minimum(mean_tpr_rf + std_tpr_rf, 1)
    tpr_lower_rf = np.maximum(mean_tpr_rf - std_tpr_rf, 0)

    # Plot mean ROC with ±1 std band
    plt.plot(
        mean_fpr,
        mean_tpr_rf,
        color="b",
        linestyle="-",
        lw=2,
        label=f"Mean ROC (AUC = {mean_auc_rf:.3f}, 95% CI = [{ci_low_rf:.3f}, {ci_high_rf:.3f}])",
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower_rf,
        tpr_upper_rf,
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
    plt.title("ROC for Tuned Random Forest with Cross-Validation")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save ROC figure
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Summarize most common best params
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
    print(f"\nFinal best parameters (mode across folds): {final_best_params}")

    # Evaluate on the last fold's test set (as in original script)
    # You could also refit a final model on the full dataset using final_best_params.
    evaluate_on_test(
        best_rf_last_fold,
        X_test_last,
        y_test_last,
        output_prefix=output_prefix,
    )

    plt.show()


def evaluate_on_test(model, X_test, y_test, output_prefix: str = "random_forest"):
    """
    Compute confusion matrix, accuracy, sensitivity, specificity,
    and 95% CIs; save results as CSV.
    """
    y_prob_rf = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred_rf = (y_prob_rf >= threshold).astype(int)

    # Confusion matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("Confusion Matrix:")
    print(cm_rf)
    print(f"Accuracy: {acc_rf:.3f}")

    # Accuracy 95% CI (normal approximation)
    n = len(y_test)
    se = np.sqrt((acc_rf * (1 - acc_rf)) / n)
    ci_low_rf = acc_rf - 1.96 * se
    ci_high_rf = acc_rf + 1.96 * se
    print(f"95% CI for accuracy: [{ci_low_rf:.3f}, {ci_high_rf:.3f}]")

    # Sensitivity & specificity
    sensitivity_rf = tp_rf / (tp_rf + fn_rf)
    specificity_rf = tn_rf / (tn_rf + fp_rf)
    print(f"Sensitivity: {sensitivity_rf:.3f}")
    print(f"Specificity: {specificity_rf:.3f}")

    # Wilson CIs for sensitivity & specificity
    sens_low_rf, sens_upp_rf = proportion_confint(
        tp_rf, tp_rf + fn_rf, alpha=0.05, method="wilson"
    )
    spec_low_rf, spec_upp_rf = proportion_confint(
        tn_rf, tn_rf + fp_rf, alpha=0.05, method="wilson"
    )
    print(f"Sensitivity 95% CI: [{sens_low_rf:.3f}, {sens_upp_rf:.3f}]")
    print(f"Specificity 95% CI: [{spec_low_rf:.3f}, {spec_upp_rf:.3f}]")

    # Binomial test: is accuracy > 0.5?
    p_value = binomtest(np.sum(y_pred_rf == y_test), n, 0.5, alternative="greater")
    print(f"P-value for accuracy > 0.5: {p_value.pvalue:.3f}")

    # Save metrics + confusion matrix
    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(cm_rf, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]).to_csv(
        cm_path
    )

    results_rf = pd.DataFrame(
        [
            {
                "model": output_prefix,
                "accuracy": acc_rf,
                "acc_ci_lower": ci_low_rf,
                "acc_ci_upper": ci_high_rf,
                "n_samples": n,
                "sensitivity": sensitivity_rf,
                "sens_ci_lower": sens_low_rf,
                "sens_ci_upper": sens_upp_rf,
                "specificity": specificity_rf,
                "spec_ci_lower": spec_low_rf,
                "spec_ci_upper": spec_upp_rf,
                "p_value_acc_gt_0.5": p_value.pvalue,
            }
        ]
    )

    metrics_path = tables_dir / f"{output_prefix}_metrics.csv"
    results_rf.to_csv(metrics_path, index=False)

    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")


def main():
    """
    Convenience entry point so you can run:

        python -m src.arvc_ml.random_forest

    Expects a file named 'dataset_knn_imputed_1.csv' in PROCESSED_DIR.
    """
    csv_path = PROCESSED_DIR / "dataset_knn_imputed_1.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {csv_path}. "
            "Place your KNN-imputed multimodal CSV there (not tracked by git)."
        )
    run_random_forest_cv(csv_path)


if __name__ == "__main__":
    main()

