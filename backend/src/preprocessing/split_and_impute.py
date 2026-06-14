import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_data(df):
    X = df.drop(columns=["F3924"])
    y = df["F3924"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Class distribution in train:")
    print(y_train.value_counts())
    print(y_train.value_counts(normalize=True) * 100)

    print("\nClass distribution in test:")
    print(y_test.value_counts())
    print(y_test.value_counts(normalize=True) * 100)

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    joblib.dump(imputer, "models/imputer.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print(f"\nFraud count in train: {y_train.sum()}")
    print(f"Fraud count in test: {y_test.sum()}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.preprocessing.clean import clean_dataset
    from src.features.engineer import engineer_features

    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)
    X_train, X_test, y_train, y_test = prepare_data(engineered_df)
