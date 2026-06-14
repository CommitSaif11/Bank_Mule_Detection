import numpy as np
from sklearn.preprocessing import LabelEncoder

MISSING_FLAG_COLS = [
    "F115", "F321", "F527", "F531", "F1692", "F2082",
    "F2122", "F2582", "F2678", "F2737", "F2956", "F3043",
]

F3889_MAP = {"G365D": 0, "L365D": 1, "L180D": 2, "L90D": 3, "L31D": 4, "L14D": 5, "L7D": 6}
F3891_MAP = {"salaried": 0, "selfemployed": 1, "student": 2, "agriculture": 3,
              "housewife": 4, "others": 5, "retired": 6}

LOG1P_COLS = ["F2678", "F3836"]


def engineer_features(df):
    df = df.copy()
    shape_before = df.shape
    new_cols = []

    # 1. Missing-value flag columns (before imputation)
    for col in MISSING_FLAG_COLS:
        if col in df.columns:
            flag_col = f"{col}_missing"
            df[flag_col] = df[col].isnull().astype(int)
            new_cols.append(flag_col)

    # 2. Encode F3889
    if "F3889" in df.columns:
        df["F3889"] = df["F3889"].map(F3889_MAP)

    # 3. Encode F3891
    if "F3891" in df.columns:
        df["F3891"] = df["F3891"].map(F3891_MAP)

    # 4. Label encode any other remaining object columns
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 5. log1p transform skewed columns
    for col in LOG1P_COLS:
        if col in df.columns:
            new_col = f"{col}_log"
            df[new_col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
            df = df.drop(columns=[col])
            new_cols.append(new_col)

    shape_after = df.shape

    print(f"Shape before: {shape_before}")
    print(f"Shape after: {shape_after}")
    print(f"New columns added: {new_cols}")

    return df


if __name__ == "__main__":
    import pandas as pd
    from src.preprocessing.clean import clean_dataset

    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)
    print(engineered_df.head())
