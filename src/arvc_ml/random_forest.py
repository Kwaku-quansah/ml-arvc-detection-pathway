# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Create empty lists to store ROC curve data for each fold
all_fpr_rf = []
all_tpr_rf = []
auc_scores_rf = []
mean_fpr = np.linspace(0, 1, 100)
best_params_list = []

# Initialize the Stratified KFold cross-validation (5-fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Perform Stratified Cross-Validation with hyperparameter tuning and plot ROC curve for each fold
fold = 1
for train_idx, test_idx in cv.split(X_scaled, y):
    # Split the data into training and testing sets for the current fold
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Define the scoring metric (ROC AUC)
    roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        scoring=roc_auc_scorer,
        cv=3,  # Use 3-fold CV within each training fold
        n_jobs=-1,  # Use all available cores
        verbose=0
    )
    
    # Perform grid search on the training data
    grid_search.fit(X_train, y_train)
    
    # Get the best parameters and model
    best_params = grid_search.best_params_
    best_params_list.append(best_params)
    print(f"Fold {fold} Best Parameters: {best_params}")
    
    # Use the best model for predictions
    best_rf = grid_search.best_estimator_
    
    # Get the predicted probabilities for the positive class
    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve for the current fold
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    auc_scores_rf.append(roc_auc_rf) #stores each fold's AUC
    
    # Interpolate to the same number of points for mean ROC calculation
    mean_tpr_rf = np.interp(mean_fpr, fpr_rf, tpr_rf)
    mean_tpr_rf[0] = 0.0  # Ensure the first point is (0,0)
    
    # Store the FPR, TPR values for the current fold
    all_fpr_rf.append(fpr_rf)
    all_tpr_rf.append(mean_tpr_rf)
    
    # Plot ROC curve for the current fold
    plt.plot(fpr_rf, tpr_rf, lw=1, alpha=0.7, label=f'Fold {fold} ROC (AUC = {roc_auc_rf:.3f})')
    fold += 1

# Calculate the mean ROC curve and AUC
mean_tpr_rf = np.mean(all_tpr_rf, axis=0)
mean_auc_rf = auc(mean_fpr, mean_tpr_rf)

# Calculate mean and 95% confidence interval for AUC
mean_auc_rf = np.mean(auc_scores_rf)
std_auc_rf = np.std(auc_scores_rf, ddof=1)
n_folds = len(auc_scores_rf)
se_auc_rf = std_auc_rf / np.sqrt(n_folds)
ci_low_rf, ci_high_rf = stats.t.interval(0.95, df=n_folds - 1, loc=mean_auc_rf, scale=se_auc_rf)

# Calculate standard deviation for the ROC curve
std_tpr_rf = np.std(all_tpr_rf, axis=0)
tpr_upper_rf = np.minimum(mean_tpr_rf + std_tpr_rf, 1)
tpr_lower_rf = np.maximum(mean_tpr_rf - std_tpr_rf, 0)

# Plot the mean ROC curve with confidence interval
plt.plot(mean_fpr, mean_tpr_rf, color='b', linestyle='-', lw=2,
         label=f'Mean ROC (AUC = {mean_auc_rf:.3f}, 95% CI = [{ci_low_rf:.3f}, {ci_high_rf:.3f}])')
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2,
                 label=f'±1 std. dev.')

# Plot the diagonal line (random classifier) for reference
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# Set the plot limits and labels
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Tuned Random Forest with Cross-Validation on Test Data')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('Hypertuned_Random_Forest_ROC.svg', format='svg')

# Summarize the most common best parameters across all folds
from collections import Counter

print("\nSummary of best parameters across all folds:")
for param in param_grid.keys():
    param_values = [fold_params[param] for fold_params in best_params_list]
    most_common = Counter(param_values).most_common(1)[0]
    print(f"{param}: most common value = {most_common[0]} (appeared in {most_common[1]}/{len(best_params_list)} folds)")

# Create a final model with the most common best parameters
final_best_params = {param: Counter([fold_params[param] for fold_params in best_params_list]).most_common(1)[0][0] 
                     for param in param_grid.keys()}
print(f"\nFinal best parameters: {final_best_params}")

plt.show()

y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred_rf = (y_prob_rf >= threshold).astype(int)  # convert to class labels

"""
Hyperparameter-tuned Random Forest multimodal model for ARVC.

- Uses 5-fold Stratified CV with GridSearchCV
- Computes fold-wise ROC curves + mean ROC with 95% CI
- Computes confusion matrix, accuracy, sensitivity, specificity + 95% CIs
- Expects the dataset to be loaded via load_knn_imputed_multimodal() in data.py
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

from .data import load_knn_imputed_multimodal
from .config import REPORTS_DIR


# Same hyperparameter grid as your notebook
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}


def run_random_forest_cv(
    n_splits: int = 5,
    random_state: int = 42,
    output_prefix: str = "random_forest",
):
    """
    Run stratified K-fold CV with hyperparameter tuning and plot ROC curves.
    """

    # 🔁 This line replaces your entire block of:
    # df = pd.read_csv(...), data/target, X, y, scaler, X_scaled, etc.
    X_scaled, y, feature_names = load_knn_imputed_multimodal()

    # Containers for ROC and AUC across folds
    all_tpr_rf = []
    auc_scores_rf = []
    mean_fpr = np.linspace(0, 1, 100)
    best_params_list = []

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold = 1
    best_rf_last_fold = None
    X_test_last = None
    y_test_last = None

    plt.figure(figsize=(8, 6))

    for train_idx, test_idx in cv.split(X_scaled, y):
        # Split the data into training and testing sets for the current fold
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Define the scoring metric (ROC AUC)
        roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True)

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=random_state),
            param_grid=PARAM_GRID,
            scoring=roc_auc_scorer,
            cv=3,  # Use 3-fold CV within each training fold
            n_jobs=-1,
            verbose=0,
        )

        # Perform grid search on the training data
        grid_search.fit(X_train, y_train)

        # Get the best parameters and model
        best_params = grid_search.best_params_
        best_params_list.append(best_params)
        print(f"Fold {fold} Best Parameters: {best_params}")

        # Use the best model for predictions
        best_rf = grid_search.best_estimator_
        best_rf_last_fold = best_rf
        X_test_last = X_test
        y_test_last = y_test

        # Get the predicted probabilities for the positive class
        y_prob_rf = best_rf.predict_proba(X_test)[:, 1]

        # Calculate ROC curve for the current fold
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        auc_scores_rf.append(roc_auc_rf)  # stores each fold's AUC

        # Interpolate to the same number of points for mean ROC calculation
        mean_tpr_rf = np.interp(mean_fpr, fpr_rf, tpr_rf)
        mean_tpr_rf[0] = 0.0  # Ensure the first point is (0,0)

        # Store the TPR values for the current fold
        all_tpr_rf.append(mean_tpr_rf)

        # Plot ROC curve for the current fold
        plt.plot(
            fpr_rf,
            tpr_rf,
            lw=1,
            alpha=0.7,
            label=f"Fold {fold} ROC (AUC = {roc_auc_rf:.3f})",
        )
        fold += 1

    # Calculate the mean ROC curve and AUC
    mean_tpr_rf = np.mean(all_tpr_rf, axis=0)
    mean_auc_rf = auc(mean_fpr, mean_tpr_rf)

    # Calculate mean and 95% confidence interval for AUC
    mean_auc_rf = np.mean(auc_scores_rf)
    std_auc_rf = np.std(auc_scores_rf, ddof=1)
    n_folds = len(auc_scores_rf)
    se_auc_rf = std_auc_rf / np.sqrt(n_folds)
    ci_low_rf, ci_high_rf = stats.t.interval(
        0.95, df=n_folds - 1, loc=mean_auc_rf, scale=se_auc_rf
    )

    # Calculate standard deviation for the ROC curve
    std_tpr_rf = np.std(all_tpr_rf, axis=0)
    tpr_upper_rf = np.minimum(mean_tpr_rf + std_tpr_rf, 1)
    tpr_lower_rf = np.maximum(mean_tpr_rf - std_tpr_rf, 0)

    # Plot the mean ROC curve with confidence interval
    plt.plot(
        mean_fpr,
        mean_tpr_rf,
        color="b",
        linestyle="-",
        lw=2,
        label=(
            f"Mean ROC (AUC = {mean_auc_rf:.3f}, "
            f"95% CI = [{ci_low_rf:.3f}, {ci_high_rf:.3f}])"
        ),
    )
    plt.fill_between(
        mean_fpr,
        tpr_lower_rf,
        tpr_upper_rf,
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
    plt.title("ROC for Tuned Random Forest with Cross-Validation")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    # Save ROC figure under reports/figures/
    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{output_prefix}_roc.svg"
    plt.savefig(roc_path, format="svg")
    print(f"ROC figure saved to {roc_path}")

    # Summarize the most common best parameters across all folds
    print("\nSummary of best parameters across all folds:")
    for param in PARAM_GRID.keys():
        param_values = [fold_params[param] for fold_params in best_params_list]
        most_common = Counter(param_values).most_common(1)[0]
        print(
            f"{param}: most common value = {most_common[0]} "
            f"(appeared in {most_common[1]}/{len(best_params_list)} folds)"
        )

    # Evaluate on the last fold's test set (similar to your original final block)
    if best_rf_last_fold is not None:
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
    and 95% CIs; save results as CSVs under reports/tables/.
    """
    y_prob_rf = model.predict_proba(X_test)[:, 1]
    threshold = 0.5
    y_pred_rf = (y_prob_rf >= threshold).astype(int)

    # Compute confusion matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("Confusion Matrix:")
    print(cm_rf)

    tn_rf, fp_rf, fn_rf, tp_rf = cm_rf.ravel()

    # Compute accuracy
    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Accuracy: {acc_rf:.3f}")

    # 95% CI for accuracy using normal approximation
    n = len(y_test)
    se = np.sqrt((acc_rf * (1 - acc_rf)) / n)
    ci_low_rf = acc_rf - 1.96 * se
    ci_high_rf = acc_rf + 1.96 * se
    print(f"95% Confidence interval for accuracy: [{ci_low_rf:.3f}, {ci_high_rf:.3f}]")

    # Sensitivity and Specificity
    sensitivity_rf = tp_rf / (tp_rf + fn_rf)
    specificity_rf = tn_rf / (tn_rf + fp_rf)
    print(f"Sensitivity: {sensitivity_rf:.3f}")
    print(f"Specificity: {specificity_rf:.3f}")

    # 95% CI for Sensitivity and Specificity (Wilson score interval)
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

    tables_dir = REPORTS_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Save confusion matrix
    cm_path = tables_dir / f"{output_prefix}_confusion_matrix.csv"
    pd.DataFrame(
        cm_rf, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]
    ).to_csv(cm_path)

    # Save metrics
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
    Convenience entry point to run from the command line:

        python -m src.arvc_ml.random_forest
    """
    run_random_forest_cv()


if __name__ == "__main__":
    main()

