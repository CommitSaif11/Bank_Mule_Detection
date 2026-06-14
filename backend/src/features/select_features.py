import joblib
import pandas as pd
from lightgbm import LGBMClassifier

BANK_LISTED_FEATURES = [
    "F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
    "F2582", "F2678_log", "F2737", "F2956", "F3043", "F3836_log",
    "F3887", "F3889", "F3891", "F3894",
]


def select_features(X_train, y_train, X_test):
    model = LGBMClassifier(
        n_estimators=200,
        scale_pos_weight=111,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False)

    top_150 = importances.head(150).index.tolist()

    must_include = []
    for col in BANK_LISTED_FEATURES:
        must_include.append(col)
        missing_col = f"{col}_missing"
        if missing_col in X_train.columns:
            must_include.append(missing_col)

    added_from_bank_list = [c for c in must_include if c in X_train.columns and c not in top_150]

    feature_names = top_150 + added_from_bank_list

    print(f"Total features selected: {len(feature_names)}")
    print(f"From top 150: {len(top_150)}")
    print(f"Added from bank list: {len(added_from_bank_list)}")

    joblib.dump(feature_names, "models/selected_features.pkl")

    X_train_selected = X_train[feature_names]
    X_test_selected = X_test[feature_names]

    return X_train_selected, X_test_selected, feature_names


if __name__ == "__main__":
    from src.preprocessing.clean import clean_dataset
    from src.features.engineer import engineer_features
    from src.preprocessing.split_and_impute import prepare_data

    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)
    X_train, X_test, y_train, y_test = prepare_data(engineered_df)

    X_train_sel, X_test_sel, feature_names = select_features(X_train, y_train, X_test)
    print(f"\nX_train_selected shape: {X_train_sel.shape}")
    print(f"X_test_selected shape: {X_test_sel.shape}")
    print(f"\nFeature names (first 20): {feature_names[:20]}")
