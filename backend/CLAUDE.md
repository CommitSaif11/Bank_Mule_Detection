# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Mule account fraud detection: a binary classification pipeline (`F3924` target, ~0.9% fraud rate) trained on `data/DataSet.csv` (9082 rows × ~3925 raw columns, prefixed `F<n>`).

## Environment

- Python 3.10 venv at `.venv` (created with `uv venv --python 3.10 .venv`). Use `.venv/Scripts/python` for all commands on Windows.
- Install deps: `uv pip install -r requirements.txt --python .venv/Scripts/python.exe`
- Run module scripts from the `backend/` root using `-m` so `src` resolves as a package, e.g.:
  ```
  .venv/Scripts/python -m src.models.train
  ```
- Standalone scripts in `notebooks/` add `backend/` to `sys.path` manually and reference data via `../data/...` or absolute paths derived from `__file__`.

## Pipeline architecture

The pipeline is a strict ordered chain — each stage's output feeds the next, and order matters (especially missing-flag creation before imputation, and fitting imputer/scaler only on train):

1. **`src/preprocessing/clean.py` — `clean_dataset(df)`**
   - First drops `BLACKLIST` columns (currently `F3912`, `F2230` — both found to be **target leakage**: near-perfect single-feature predictors of `F3924`). Any new leakage column discovered must be added here first.
   - Then drops fully-null columns, columns >80% null, and zero-variance numeric columns.
   - `PRESERVE_COLS` (categorical columns F2230/F3886-F3893 area) are exempt from the drop rules — note F2230 is exempt from drop-rules but still removed via BLACKLIST.
   - Returns `(cleaned_df, log_dict)` where log_dict records what was dropped and `final_shape`.

2. **`src/features/engineer.py` — `engineer_features(df)`**
   - Creates `<col>_missing` binary flags for key numeric columns **before** imputation happens downstream.
   - Label-encodes `F3889` and `F3891` via fixed maps; any other object columns get `LabelEncoder`.
   - Log1p-transforms `F2678`/`F3836` (sign-aware: `sign(x) * log1p(|x|)`) into `F2678_log`/`F3836_log`, dropping originals.

3. **`src/preprocessing/split_and_impute.py` — `prepare_data(df)`**
   - Stratified 80/20 train/test split (`random_state=42`).
   - `SimpleImputer(strategy='median')` and `StandardScaler` fit on **train only**, applied to both.
   - Saves `models/imputer.pkl`, `models/scaler.pkl`.

4. **`src/features/select_features.py` — `select_features(X_train, y_train, X_test)`**
   - Trains a quick LightGBM (`scale_pos_weight=111`) to rank feature importance.
   - Selects top 150 by importance, then force-includes a hardcoded `BANK_LISTED_FEATURES` list (+ their `_missing` flags) if not already present.
   - Saves `models/selected_features.pkl` (list of column names).

5. **`src/models/train.py`**
   - Runs the full chain above, applies `SMOTE(random_state=42, k_neighbors=5)` to the train set only, then trains LightGBM / XGBoost / RandomForest with fixed hyperparameters.
   - Evaluates on the untouched test set (confusion matrix, classification report, AUC-ROC, AUC-PR).
   - Saves `models/lgbm_model.pkl`, `models/xgb_model.pkl`, `models/rf_model.pkl`.

## Known issue: data leakage

This dataset has had multiple leakage columns discovered via correlation analysis + depth-1 decision tree splits (a single-feature tree achieving near-100% recall/accuracy on the fraud class is the smoking-gun test — see `notebooks/06_find_remaining_leak.py` and `notebooks/07_post_fix_check.py` for the methodology). When evaluating any new feature or column for leakage, reuse this depth-1-tree + correlation approach rather than just looking at overall model metrics — perfect/near-perfect AUC with **zero variance across CV folds** is the signature of leakage, not real performance.

## Notebooks

`notebooks/0N_*.py` are sequential, runnable investigation/EDA scripts (not actual `.ipynb` except `01_data_exploration.ipynb`). They're exploratory artifacts of the leakage-investigation process and write plots to `reports/`.
