# ARVC ML Detection Pathway

Machine learning models for Arrhythmogenic Right Ventricular Cardiomyopathy (ARVC) detection using a multimodal feature set.

## Project Goals

- Implement a reproducible ML pipeline for ARVC diagnosis.
- Compare eight supervised learning algorithms on a multimodal feature set.
- Perform statistical comparison of models (Friedman test, Nemenyi post-hoc).
- Evaluate multimodal and ECG-only variants for Gradient Boosted Trees.
- Provide interpretable outputs aligned with Task Force Criteria for ARVC diagnosis.

## Repository Layout

- `src/arvc_ml/`
  - `config.py` – Global settings (paths, random seeds, target column).
  - `data.py` – Data loading and preprocessing helpers (expects local CSV, not in repo).
  - `random_forest.py` – Hyperparameter-tuned Random Forest model.
  - `gbt.py` – Gradient Boosted Trees model (to be implemented).
  - `logistic_regression.py` – Logistic Regression model (stub).
  - `naive_bayes.py` – Naive Bayes model (stub).
  - `ols.py` – OLS-based model (stub).
  - `lasso_logistic.py` – Lasso Logistic Regression model (stub).
  - `tabnet.py` – TabNet model (stub).
  - `decision_tree.py` – Decision Tree model (stub).
  
#This repository does not include any clinical data. All data loading is handled via stub functions in src/arvc_ml/data.py that must be implemented locally with de-identified datasets

> **Note:** Clinical data are **not** included in this repository. Scripts expect a local file (e.g., `dataset_knn_imputed_1.csv`) placed in the project root and ignored by git.
