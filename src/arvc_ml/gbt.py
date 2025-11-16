"""
Gradient Boosted Trees (GBT) models for ARVC:

- Multimodal GBT with hyperparameter tuning (RandomizedSearchCV or GridSearchCV)
- ECG-only GBT with the same architecture
- ROC curves per fold + mean ROC with 95% CI
- Confusion matrix, accuracy, sensitivity, specificity + 95% CIs
- Permutation feature importance for the multimodal model

Data loading and scaling is handled by:

    load_knn_imputed_multimodal()
    load_knn_imputed_ecg_only()

in data.py.
"""

from typing import Callable, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

from scipy import stats
from scipy.stats import randint, uniform, binomtest
from statsmodels.stats.proportion import proportion_confint

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    accuracy_score,
    make_scorer,
)
from sklearn.inspection import permutation_importance

from .data import (
    load_knn_imputed_multimodal,
    load_knn_imputed_ecg_only,
)
from .config import REPORTS_DIR, RANDOM_STATE


# -----------------------------
# Hyperparameter search spaces
# -----------------------------

# Grid for GridSearchCV
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 4, 5, 6],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.8, 0.9, 1.0],
    "max_features": ["sqrt", "log2", None],
}

# Distributions for RandomizedSearchCV (faster for large spaces)
PARAM_DIST = {
    "n_estimators": randint(50, 300),
    "learning_rate": uniform(0.01, 0.2),
    "max_depth": randint(3, 10),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "subsample": uniform(0.7, 0.3),
    "max_features": ["sqrt", "log2", None],
}

# Toggle between grid search and randomized search
USE_GRID_SEARCH = False  # set to True if you prefer GridSearchCV


# -----------------------------
# Core CV runner
# -----------------------------

def _run_gbt_cv(
    loader_fn: Callable[[], Tuple[np.ndarray, np.ndarray, list]],
    output_prefix: str,
    title: str,
    do_permutation_importance: bool = False,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
):
    """
    Generic runner for GBT with nested CV and hyperparameter tuning.

    Parameters
    ----------
    loader_fn : callable
        Function that returns (X_scaled, y, feature_names).
    output_prefix : str
        Prefix for saving figures and tables (e.g., 'gbt_multimodal').
    title : str
        Title for ROC plot.
    do_permutation_importance : bool
        Whether to compute permutation feature importance on the last test fold.
    n_splits : int
        Number of outer CV folds.
    random_state : int
        Random seed.
    """

    # Load data
    X_scaled, y, feature_names = loader_fn()

    all_tpr = []
    auc_scores = []
    mean_fpr = np.linspace(0, 1, 100)
    best_params_list = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    last_best_model = None
    last_X_test = None
    last_y_test = None

    plt.figure(figsize=(8, 6))

    for train_idx, test_idx in cv.split(X_scaled, y):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ROC AUC as scoring metric
        roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)

        if USE_GRID_SEARCH:
            print(f"Fold {fold}: Running GridSearchCV (this might take a while)...")
            search = GridSearchCV(
                estimator=GradientBoostingClassifier(random_state=random_state),
                param_grid=PARAM_GRID,
                scoring=roc_auc_scorer,
                cv=3,
                n_jobs=-1,
                verbose=1,
            )
            param_keys = list(PARAM_GRID.keys())
        else:
            print(f"Fold {fold}: Running RandomizedSearchCV...")
            search = RandomizedSearchCV(
                estimator=GradientBoostingClassifier(random_state=random_state),
                param_distributions=PARAM_DIST,
                n_iter=20,
                scoring=roc_auc_scorer,
                cv=3,
                n_jobs=-1,
                verbose=1,
                random_state=random_state,
            )
            param_keys = list(PARAM_DIST.keys())

        search.fit(X_train, y_train)

        best_params = search.best_params_
        best_params_list.append(best_params)
        print(f"Fold {fold} Best Parameters: {best_params}")
        print(f"Fold {fold} Best CV Score: {search.best_score_:.3f}")

        best_gb = search.best_estimator_

        # Track last fold's best model + test data
        last_best_model = best_gb
        last_X_test = X_test
        last_y_test = y_test

        # ROC for this fold
        y_prob = best_gb.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

        mean_tpr = np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        all_tpr.append(mean_tpr)

        plt.plot(
            fpr,
            tpr,
            lw=1.2,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc:.3f})",
        )

        fold += 1

    # AUC summary across folds
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores, ddof=1)
    n_folds = len(auc_scores)
    se_auc = std_auc / np.sqrt(n_folds)
    ci_low, ci_high = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc, scale=se_auc
    )

    # Mean ROC curve
    mean_tpr = np.mean(all_tpr, axis=0)
    mean_auc_curve = auc(mean_fpr, mean_tpr)

    std_tpr = np.std(all_tpr, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        linestyle="-",
        lw=2,
        label=(
            f"Mean ROC (AUC = {mean_auc_curve:.3f}, "
            f"95% CI = [{ci_low:.3f}, {ci_high:.3f}])"
        ),
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower,
        tpr_upper,
        color="blue",
        alpha=0.2,
        label="±1 std. dev.",
    )

    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save ROC figure
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Summary of best parameters across folds
    print("\nSummary of best parameters across all folds:")
    for param in param_keys:
        param_values = [fold_params.get(param, None) for fold_params in best_params_list]
        counts = Counter(param_values)
        most_common = counts.most_common(1)[0]
        print(
            f"{param}: most common value = {most_common[0]} "
            f"(appeared in {most_common[1]}/{len(best_params_list)} folds)"
        )

    final_best_params = {
        param: Counter(
            [fold_params.get(param, None) for fold_params in best_params_list]
        ).most_common(1)[0][0]
        for param in param_keys
    }
    print(f"\nFinal recommended parameters: {final_best_params}")

    # Evaluate on the last fold's test split
    if last_best_model is not None and last_X_test is not None:
        evaluate_on_test(
            last_best_model,
            last_X_test,
            last_y_test,
            output_prefix=output_prefix,
        )

        if do_permutation_importance:
            compute_permutation_importance(
                last_best_model,
                last_X_test,
                last_y_test,
                feature_names,
                output_prefix=output_prefix,
            )

    plt.show()


# -----------------------------
# Permutation importance
# -----------------------------

def compute_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    output_prefix: str = "gbt_multimodal",
):
    """
    Compute permutation feature importance on a held-out test set and save
    results + boxplot.

    Mirrors your original permutation importance logic but writes to
    reports/ subfolders.
    """

    print("\nComputing permutation feature importance...")
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    sorted_idx = np.argsort(result.importances_mean)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]

    perm_importance_df = pd.DataFrame(
        {
            "feature": sorted_features,
            "mean_importance": result.importances_mean[sorted_idx],
            "std_importance": result.importances_std[sorted_idx],
        }
    )

    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    perm_path = tables_dir / f"{output_prefix}_permutation_importance.csv"
    perm_importance_df.to_csv(perm_path, index=False)
    print(f"Permutation importance table saved to {perm_path}")

    # Boxplot (like your original importances boxplot)
    top_n = min(15, len(sorted_features))
    importances = pd.DataFrame(
        result.importances[sorted_idx].T,
        columns=sorted_features,
    )

    ax = importances.iloc[:, :top_n].plot.box(
        vert=False,
        whis=10,
        figsize=(10, 6),
    )
    ax.axvline(x=0, color="k", linestyle="--")
    ax.set_xlabel("Decrease in Accuracy Score")
    ax.set_title("Permutation Importance Variability (Test Data)")
    plt.tight_layout()

    box_path = figures_dir / f"{output_prefix}_permutation_box.svg"
    plt.savefig(box_path, dpi=600)
    plt.show()

    print("✅ Permutation importance plot saved.")


# -----------------------------
# Evaluation on test fold
# -----------------------------

def evaluate_on_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_prefix: str = "gbt",
):
    """
    Compute confusion matrix, accuracy, sensitivity, specificity,
    and 95% CIs; save results as CSVs.
    """

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")

    n = len(y_test)
    se = np.sqrt((acc * (1 - acc)) / n)
    ci_low = acc - 1.96 * se
    ci_high = acc + 1.96 * se
    print(f"95% Confidence interval for accuracy: [{ci_low:.3f}, {ci_high:.3f}]")

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")

    sens_low, sens_upp = proportion_confint(
        tp, tp + fn, alpha=0.05, method="wilson"
    )
    spec_low, spec_upp = proportion_confint(
        tn, tn + fp, alpha=0.05, method="wilson"
    )
    print(f"Sensitivity 95% CI: [{sens_low:.3f}, {sens_upp:.3f}]")
    print(f"Specificity 95% CI: [{spec_low:.3f}, {spec_upp:.3f}]")

    p_value = binomtest((y_pred == y_test).sum(), n, 0.5, alternative="greater")
    print(f"P-value for accuracy > 0.5: {p_value.pvalue:.3f}")

    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(
        cm,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"],
    ).to_csv(cm_path)

    results_df = pd.DataFrame(
        [
            {
                "model": output_prefix,
                "accuracy": acc,
                "acc_ci_lower": ci_low,
                "acc_ci_upper": ci_high,
                "n_samples": n,
                "sensitivity": sensitivity,
                "sens_ci_lower": sens_low,
                "sens_ci_upper": sens_upp,
                "specificity": specificity,
                "spec_ci_lower": spec_low,
                "spec_ci_upper": spec_upp,
                "p_value_acc_gt_0.5": p_value.pvalue,
            }
        ]
    )
    metrics_path = tables_dir / f"{output_prefix}_metrics.csv"
    results_df.to_csv(metrics_path, index=False)

    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")


# -----------------------------
# Public entry points
# -----------------------------

def run_gbt_multimodal_cv():
    """
    Run multimodal GBT CV with hyperparameter tuning and permutation importance.
    """
    _run_gbt_cv(
        loader_fn=load_knn_imputed_multimodal,
        output_prefix="gbt_multimodal",
        title="ROC for Tuned Gradient Boosted Trees (Multimodal)",
        do_permutation_importance=True,
    )


def run_gbt_ecg_only_cv():
    """
    Run ECG-only GBT CV with hyperparameter tuning.
    """
    _run_gbt_cv(
        loader_fn=load_knn_imputed_ecg_only,
        output_prefix="gbt_ecg_only",
        title="ROC for Tuned Gradient Boosted Trees (ECG-only)",
        do_permutation_importance=False,
    )


def main():
    """
    Convenience entry point.

    You can change this to run either multimodal or ECG-only by default.
    """
    # Default: multimodal
    run_gbt_multimodal_cv()
    # If you want to also run ECG-only in the same call, uncomment:
    # run_gbt_ecg_only_cv()


if __name__ == "__main__":
    main()

