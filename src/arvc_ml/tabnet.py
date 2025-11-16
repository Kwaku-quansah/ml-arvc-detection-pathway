"""
Hyperparameter-tuned TabNet multimodal model for ARVC.

- 5-fold Stratified CV
- Inner 3-fold split inside each outer fold for hyperparameter tuning
- Manual grid search over TabNet parameters
- ROC curves per fold + mean ROC with 95% CI
- Confusion matrix, accuracy, sensitivity, specificity + 95% CIs

Data loading and scaling is handled by load_knn_imputed_multimodal() in data.py.

Note: Requires pytorch-tabnet, torch, and tqdm to be installed.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product

from tqdm.auto import tqdm

from scipy import stats
from scipy.stats import binomtest
from statsmodels.stats.proportion import proportion_confint

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    accuracy_score,
)
from sklearn.model_selection import StratifiedKFold

import torch
from torch import optim
from pytorch_tabnet.tab_model import TabNetClassifier

from .data import load_knn_imputed_multimodal
from .config import REPORTS_DIR, RANDOM_STATE


# Same parameter grid as in your notebook
PARAM_GRID = {
    "n_d": [8, 16, 24],
    "n_a": [8, 16, 24],
    "n_steps": [3, 5, 7],
    "gamma": [1.0, 1.5, 2.0],
    "learning_rate": [0.01, 0.02, 0.005],
    "lambda_sparse": [0, 0.001, 0.01],
}


def evaluate_params(X_train, y_train, X_val, y_val, params):
    """
    Train a TabNetClassifier with given params and return validation AUC and model.
    """

    model = TabNetClassifier(
        n_d=params["n_d"],
        n_a=params["n_a"],
        n_steps=params["n_steps"],
        gamma=params["gamma"],
        lambda_sparse=params["lambda_sparse"],
        optimizer_fn=optim.Adam,
        optimizer_params={"lr": params["learning_rate"]},
        mask_type="entmax",
        seed=RANDOM_STATE,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        max_epochs=50,  # reduced for faster tuning, as in your notebook
        patience=5,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        verbose=0,
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    val_auc = auc(fpr, tpr)

    return val_auc, model


def run_tabnet_cv(
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
    output_prefix: str = "tabnet",
):
    """
    Outer stratified K-fold CV with inner validation split for TabNet hyperparameter tuning.
    """

    # 🔁 Load data via shared loader (replaces read_csv + manual scaling)
    X_scaled, y, feature_names = load_knn_imputed_multimodal()

    all_tpr_tb = []
    auc_scores_tb = []
    mean_fpr = np.linspace(0, 1, 100)
    best_params_list = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    final_model_last_fold = None
    X_test_last = None
    y_test_last = None

    plt.figure(figsize=(8, 6))

    for train_idx, test_idx in cv.split(X_scaled, y):
        print(f"\nProcessing Fold {fold}")

        # Outer split: train_full / test
        X_train_full, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train_full, y_test = y[train_idx], y[test_idx]

        # Inner split: create a validation set from X_train_full
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        inner_train_idx, val_idx = next(inner_cv.split(X_train_full, y_train_full))

        X_train, X_val = X_train_full[inner_train_idx], X_train_full[val_idx]
        y_train, y_val = y_train_full[inner_train_idx], y_train_full[val_idx]

        # Manual grid search over PARAM_GRID on (X_train, X_val)
        best_auc = 0.0
        best_params = None

        # Generate all parameter combinations
        param_combinations = list(
            product(
                PARAM_GRID["n_d"],
                PARAM_GRID["n_a"],
                PARAM_GRID["n_steps"],
                PARAM_GRID["gamma"],
                PARAM_GRID["learning_rate"],
                PARAM_GRID["lambda_sparse"],
            )
        )
        print(f"Evaluating {len(param_combinations)} parameter combinations...")

        for comb in tqdm(param_combinations):
            param_dict = {
                "n_d": comb[0],
                "n_a": comb[1],
                "n_steps": comb[2],
                "gamma": comb[3],
                "learning_rate": comb[4],
                "lambda_sparse": comb[5],
            }

            val_auc, _ = evaluate_params(X_train, y_train, X_val, y_val, param_dict)

            if val_auc > best_auc:
                best_auc = val_auc
                best_params = param_dict

        print(f"Best validation AUC (fold {fold}): {best_auc:.4f}")
        print(f"Best parameters (fold {fold}): {best_params}")
        best_params_list.append(best_params)

        # Train final model with best params on the full training set (X_train_full)
        final_model = TabNetClassifier(
            n_d=best_params["n_d"],
            n_a=best_params["n_a"],
            n_steps=best_params["n_steps"],
            gamma=best_params["gamma"],
            lambda_sparse=best_params["lambda_sparse"],
            optimizer_fn=optim.Adam,
            optimizer_params={"lr": best_params["learning_rate"]},
            mask_type="entmax",
            seed=random_state,
        )

        final_model.fit(
            X_train_full,
            y_train_full,
            max_epochs=100,
            patience=10,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            verbose=0,
        )

        final_model_last_fold = final_model
        X_test_last = X_test
        y_test_last = y_test

        # ROC on outer test set
        y_prob_tb = final_model.predict_proba(X_test)[:, 1]
        fpr_tb, tpr_tb, _ = roc_curve(y_test, y_prob_tb)
        roc_auc_tb = auc(fpr_tb, tpr_tb)
        auc_scores_tb.append(roc_auc_tb)

        mean_tpr_tb = np.interp(mean_fpr, fpr_tb, tpr_tb)
        mean_tpr_tb[0] = 0.0
        all_tpr_tb.append(mean_tpr_tb)

        plt.plot(
            fpr_tb,
            tpr_tb,
            lw=1.0,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc_tb:.3f})",
        )

        fold += 1

    # Mean ROC curve across folds
    mean_tpr_tb = np.mean(all_tpr_tb, axis=0)
    mean_auc_tb = auc(mean_fpr, mean_tpr_tb)

    mean_auc_tb = np.mean(auc_scores_tb)
    std_auc_tb = np.std(auc_scores_tb, ddof=1)
    n_folds = len(auc_scores_tb)
    se_auc_tb = std_auc_tb / np.sqrt(n_folds)
    ci_low_tb, ci_high_tb = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc_tb, scale=se_auc_tb
    )

    std_tpr_tb = np.std(all_tpr_tb, axis=0)
    tpr_upper_tb = np.minimum(mean_tpr_tb + std_tpr_tb, 1)
    tpr_lower_tb = np.maximum(mean_tpr_tb - std_tpr_tb, 0)

    plt.plot(
        mean_fpr,
        mean_tpr_tb,
        color="b",
        linestyle="-",
        lw=2,
        label=(
            f"Mean ROC (AUC = {mean_auc_tb:.3f}, "
            f"95% CI = [{ci_low_tb:.3f}, {ci_high_tb:.3f}])"
        ),
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower_tb,
        tpr_upper_tb,
        color="blue",
        alpha=0.2,
        label="±1 std. dev.",
    )

    plt.plot([0, 1], [0, 1], color="gray", lw=2, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC for Tuned TabNet with Cross-Validation")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Summarize best parameters across folds
    from collections import Counter

    print("\nSummary of best parameters across all folds:")
    for param in PARAM_GRID.keys():
        param_values = [fold_params[param] for fold_params in best_params_list]
        value_counts = Counter(param_values)
        most_common = value_counts.most_common(1)[0]
        print(
            f"{param}: most common value = {most_common[0]} "
            f"(appeared in {most_common[1]}/{len(best_params_list)} folds)"
        )
        for value, count in value_counts.items():
            print(f"  - {value}: {count} folds")

    final_best_params = {
        param: Counter([fold_params[param] for fold_params in best_params_list])
        .most_common(1)[0][0]
        for param in PARAM_GRID.keys()
    }
    print(f"\nFinal recommended parameters: {final_best_params}")

    # Evaluate on the last fold's test set
    if final_model_last_fold is not None:
        evaluate_on_test(
            final_model_last_fold,
            X_test_last,
            y_test_last,
            output_prefix=output_prefix,
        )

    plt.show()


def evaluate_on_test(
    model, X_test, y_test, output_prefix: str = "tabnet"
):
    """
    Compute confusion matrix, accuracy, sensitivity, specificity,
    and 95% CIs; save results as CSVs.
    """
    y_prob_tb = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred_tb = (y_prob_tb >= threshold).astype(int)

    cm_tb = confusion_matrix(y_test, y_pred_tb)
    print("Confusion Matrix:")
    print(cm_tb)

    tn_tb, fp_tb, fn_tb, tp_tb = cm_tb.ravel()

    acc_tb = accuracy_score(y_test, y_pred_tb)
    print(f"Accuracy: {acc_tb:.3f}")

    n = len(y_test)
    se = np.sqrt((acc_tb * (1 - acc_tb)) / n)
    ci_low_tb = acc_tb - 1.96 * se
    ci_high_tb = acc_tb + 1.96 * se
    print(f"95% Confidence interval for accuracy: [{ci_low_tb:.3f}, {ci_high_tb:.3f}]")

    sensitivity_tb = tp_tb / (tp_tb + fn_tb)
    specificity_tb = tn_tb / (tn_tb + fp_tb)
    print(f"Sensitivity: {sensitivity_tb:.3f}")
    print(f"Specificity: {specificity_tb:.3f}")

    sens_low_tb, sens_upp_tb = proportion_confint(
        tp_tb, tp_tb + fn_tb, alpha=0.05, method="wilson"
    )
    spec_low_tb, spec_upp_tb = proportion_confint(
        tn_tb, tn_tb + fp_tb, alpha=0.05, method="wilson"
    )
    print(f"Sensitivity 95% CI: [{sens_low_tb:.3f}, {sens_upp_tb:.3f}]")
    print(f"Specificity 95% CI: [{spec_low_tb:.3f}, {spec_upp_tb:.3f}]")

    p_value = binomtest((y_pred_tb == y_test).sum(), n, 0.5, alternative="greater")
    print(f"P-value for accuracy > 0.5: {p_value.pvalue:.4f}")

    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(
        cm_tb,
        columns=["Pred 0", "Pred 1"],
        index=["True 0", "True 1"],
    ).to_csv(cm_path)

    results_tb = pd.DataFrame(
        [
            {
                "model": output_prefix,
                "accuracy": acc_tb,
                "acc_ci_lower": ci_low_tb,
                "acc_ci_upper": ci_high_tb,
                "n_samples": n,
                "sensitivity": sensitivity_tb,
                "sens_ci_lower": sens_low_tb,
                "sens_ci_upper": sens_upp_tb,
                "specificity": specificity_tb,
                "spec_ci_lower": spec_low_tb,
                "spec_ci_upper": spec_upp_tb,
                "p_value_acc_gt_0.5": p_value.pvalue,
            }
        ]
    )
    metrics_path = tables_dir / f"{output_prefix}_metrics.csv"
    results_tb.to_csv(metrics_path, index=False)

    print(f"Confusion matrix saved to {cm_path}")
    print(f"Metrics saved to {metrics_path}")


def main():
    """
    Convenience entry point so you can run:

        python -m src.arvc_ml.tabnet
    """
    run_tabnet_cv()


if __name__ == "__main__":
    main()

