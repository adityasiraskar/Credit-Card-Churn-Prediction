# Customer Churn Prediction in the Banking Sector.

An end-to-end machine learning project for predicting credit card customer churn in
the banking sector. The project includes exploratory analysis, preprocessing, model
training with MLflow tracking, model comparison, and an interactive Streamlit
prediction app.
=

## Project Goals

This project answers two practical questions:

1. Which machine learning model predicts bank customer churn best?
2. Does SMOTE oversampling of the minority (churn) class improve prediction
   performance, and what does it trade off against precision?

The current saved app artifacts serve an **XGBoost model trained without SMOTE**
(see [Model Training Summary](#model-training-summary)).

## Dataset

Dataset: Credit Card Customers - Kaggle
https://www.kaggle.com/datasets/anwarsan/credit-card-bank-churn

Expected raw file:

```text
data/raw/BankChurners.csv
```

The raw dataset contains 10,127 customers. The target column is
`Attrition_Flag`, where `Attrited Customer` represents churn and
`Existing Customer` represents non-churn.

The raw CSV is intentionally ignored by Git because it is a local dataset file.

## Project Structure

```text
Customer-Churn-Prediction/
|-- app/
|   `-- streamlit_app.py
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
|   |-- <ModelName>_model.pkl      (one per model, saved by notebook 01)
|   `-- preprocessor.pkl           (built by scripts/build_artifacts.py)
|-- notebooks/
|   |-- 01_churn_pipeline_without_smote.ipynb (8 classifiers, no SMOTE)
|   |-- 02_churn_pipeline_using_smote.ipynb  (same 8 classifiers, with SMOTE)
|   `-- mlflow.db                            (local MLflow tracking store, gitignored)
|-- outputs/
|   `-- metrics/
|-- scripts/
|   `-- build_artifacts.py
|-- src/
|   `-- preprocessing.py
|-- tests/
|   `-- test_serving_artifacts.py
|-- .gitignore
|-- README.md
`-- requirements.txt
```

## Quick Start on Windows PowerShell

Run these commands from PowerShell.

```powershell
cd "D:\Churn Prediction\Customer-Churn-Prediction"
```

If you want to use the existing virtual environment in the parent folder:

```powershell
..\.venv-1\Scripts\Activate.ps1
```

If activation is blocked by PowerShell policy for this terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
..\.venv-1\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Build the serving preprocessor artifact:

```powershell
python scripts\build_artifacts.py
```

Run tests:

```powershell
pytest
```

Run the Streamlit app:

```powershell
streamlit run app\streamlit_app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Fresh Environment Setup

If the existing parent virtual environment is missing or you want a new one
inside the project, run:

```powershell
cd "D:\Churn Prediction\Customer-Churn-Prediction"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Then continue with:

```powershell
python scripts\build_artifacts.py
pytest
streamlit run app\streamlit_app.py
```

## Full Rebuild Workflow

Use this path if you want to regenerate processed data, model artifacts, and
metrics from the raw Kaggle dataset.

1. Place the raw dataset here:

```text
data/raw/BankChurners.csv
```

2. Start Jupyter:

```powershell
jupyter notebook
```

3. Run notebooks in this order:

```text
notebooks/01_churn_pipeline_without_smote.ipynb
notebooks/02_churn_pipeline_using_smote.ipynb
```

Notebook 01 trains and saves each model as `models/<ModelName>_model.pkl`
(e.g. `models/XGBoost_model.pkl`) and writes
`outputs/metrics/model_report_without_smote.csv`. Notebook 02 repeats the
same 8 classifiers on SMOTE-resampled training data and writes
`outputs/metrics/model_report_with_smote.csv`. Both notebooks log every
run's params, metrics, figures, and models to a local MLflow store at
`notebooks/mlflow.db`.

4. Rebuild the serving preprocessor after training:

```powershell
python scripts\build_artifacts.py
```

This fits `src/preprocessing.py`'s `ChurnPreprocessor` on the raw CSV and
aligns its output columns with the champion model's own `feature_names_in_`
(read directly from `models/XGBoost_model.pkl` — there is no separate
feature-columns file).

5. Verify and run the app:

```powershell
pytest
streamlit run app\streamlit_app.py
```

## MLflow Tracking

Both training notebooks log to a local SQLite-backed MLflow store at
`notebooks/mlflow.db`, under two experiments:

- `churn_prediction_experiment_without_smote`
- `churn_prediction_experiment_with_smote`

To browse runs, params, metrics, and logged artifacts (ROC curves, confusion
matrices, classification reports, models):

```powershell
cd notebooks
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Then open http://127.0.0.1:5000. The `mlflow ui` command must be run from the
`notebooks/` folder (or point at the same `mlflow.db` path the notebooks use),
otherwise it will open an empty store.

## What the App Uses

The Streamlit app loads:

```text
models/XGBoost_model.pkl
models/preprocessor.pkl
```

`app/streamlit_app.py` looks for `models/XGBoost_model.pkl` first (the
champion model — see [Model Training Summary](#model-training-summary)) and
only falls back to the most recently modified `models/*_model.pkl` if that
file is missing. `scripts/build_artifacts.py` fits the shared preprocessor in
`src/preprocessing.py` and aligns its feature order with the champion model's
`feature_names_in_`, so the app serves the exact feature shape the model
expects. There is no separate `feature_columns.pkl` — the notebooks never
produce one.

## Model Training Summary
Both notebooks train the same 8 classifiers:

| Model | Name |
|---|---|
| LR | Logistic Regression |
| NB | Naive Bayes |
| DT | Decision Tree Classifier |
| RF | Random Forest |
| AB | AdaBoost |
| GB | Gradient Boosting |
| XGB | XGBoost |
| LGBM | LightGBM |

Notebook 01 trains on the original (imbalanced) training split. Notebook 02
applies SMOTENC to the training split only (never to the test split, to avoid
leakage) and retrains the same 8 classifiers.

**Champion model: XGBoost without SMOTE** — 95% churn precision, 91% churn
F1-score, 97% accuracy, 14 false positives. **LightGBM with SMOTE** is a
reasonable alternative when missing a churner is costlier than a false
alarm, reaching 91% churn recall at the cost of more false positives.

## Useful Commands

Compile Python files:

```powershell
python -m compileall src scripts app tests
```

Run tests:

```powershell
pytest
```

Rebuild serving artifact:

```powershell
python scripts\build_artifacts.py
```

Run app on a specific port:

```powershell
streamlit run app\streamlit_app.py --server.port 8501
```

Remove local cache files:

```powershell
Remove-Item -Recurse -Force .pytest_cache, src\__pycache__, app\__pycache__, scripts\__pycache__, tests\__pycache__ -ErrorAction SilentlyContinue
```

Remove local MLflow files after closing Jupyter/Python processes:

```powershell
Remove-Item -Recurse -Force notebooks\mlruns, notebooks\mlflow.db -ErrorAction SilentlyContinue
```

## Git Ignore Notes

The `.gitignore` file excludes local data, generated model pickle files,
MLflow runs/databases, Python caches, notebook checkpoints, virtual
environments, environment files, and OS clutter.

Keep source code, notebooks, tests, and lightweight metrics in Git. Regenerate
large runtime artifacts locally when needed.

## Troubleshooting

If the app says the preprocessor is missing:

```powershell
python scripts\build_artifacts.py
```

If the app says the model and preprocessor feature counts do not match, retrain
or rebuild artifacts in this order:

```powershell
jupyter notebook
python scripts\build_artifacts.py
pytest
streamlit run app\streamlit_app.py
```

If `notebooks\mlflow.db` cannot be deleted, close any running Jupyter, MLflow,
or Python process first, then run:

```powershell
Remove-Item -LiteralPath "notebooks\mlflow.db" -Force
```

## License

This project is for educational and portfolio use. The Kaggle dataset is subject
to its own license terms. The reference paper is licensed under CC BY-NC 4.0.
