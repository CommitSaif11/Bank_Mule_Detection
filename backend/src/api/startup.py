from pathlib import Path

import pandas as pd


def check_models_exist():
    required_files = [
        'models/lgbm_model.pkl',
        'models/imputer.pkl',
        'models/scaler.pkl',
        'models/selected_features.pkl',
        'saved_models/risk_scores.csv',
        'saved_models/shap_values.pkl',
    ]
    return all(Path(f).exists() for f in required_files)


def startup_check():
    if not check_models_exist():
        print("Model files not found. Running pipeline...")
        from src.pipeline.run_pipeline import run_full_pipeline

        dataset_path = Path("data/DataSet.csv")
        if not dataset_path.exists():
            print(
                "ERROR: DataSet.csv not found in backend/data/\n"
                "Please add DataSet.csv to backend/data/ and rebuild."
            )
            return

        raw_df = pd.read_csv(dataset_path)
        run_full_pipeline(raw_df)

        from src.api.dependencies import clear_cache
        clear_cache()
        print("Pipeline complete. Models ready.")
    else:
        print("Models found. API ready.")
