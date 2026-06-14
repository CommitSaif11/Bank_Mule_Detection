import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features
from src.preprocessing.split_and_impute import prepare_data
from src.features.select_features import select_features


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")

    report_dict = classification_report(y_test, y_pred, output_dict=True)
    return {
        "recall": report_dict["1"]["recall"],
        "precision": report_dict["1"]["precision"],
        "f1": report_dict["1"]["f1-score"],
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
    }


if __name__ == "__main__":
    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)
    X_train, X_test, y_train, y_test = prepare_data(engineered_df)
    X_train_selected, X_test_selected, feature_names = select_features(X_train, y_train, X_test)

    print("\nClass counts before SMOTE:")
    print(y_train.value_counts())

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

    print("\nClass counts after SMOTE:")
    print(pd.Series(y_train_resampled).value_counts())

    # Model A - LightGBM
    lgbm_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(X_train_resampled, y_train_resampled)

    # Model B - XGBoost
    try:
        xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=5,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train_resampled, y_train_resampled)
    except TypeError:
        xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=5,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        xgb_model.fit(X_train_resampled, y_train_resampled)

    # Model C - Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train_resampled, y_train_resampled)

    results = {}
    results["LightGBM"] = evaluate_model("Model A - LightGBM", lgbm_model, X_test_selected, y_test)
    results["XGBoost"] = evaluate_model("Model B - XGBoost", xgb_model, X_test_selected, y_test)
    results["Random Forest"] = evaluate_model("Model C - Random Forest", rf_model, X_test_selected, y_test)

    joblib.dump(lgbm_model, "models/lgbm_model.pkl")
    joblib.dump(xgb_model, "models/xgb_model.pkl")
    joblib.dump(rf_model, "models/rf_model.pkl")

    print(f"\n{'=' * 60}")
    print("Model Comparison")
    print(f"{'=' * 60}")
    print(f"{'Model':<15} {'Recall(fraud)':<15} {'Precision(fraud)':<18} {'F1(fraud)':<12} {'AUC-ROC':<10} {'AUC-PR':<10}")
    for name, r in results.items():
        print(f"{name:<15} {r['recall']:<15.4f} {r['precision']:<18.4f} {r['f1']:<12.4f} {r['auc_roc']:<10.4f} {r['auc_pr']:<10.4f}")
