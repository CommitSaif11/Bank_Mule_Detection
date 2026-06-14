import pandas as pd

PRESERVE_COLS = [
    "F2230", "F3886", "F3887", "F3888", "F3889",
    "F3890", "F3891", "F3892", "F3893",
]

BLACKLIST = ["F3912", "F2230"]


def clean_dataset(df):
    df = df.copy()

    df = df.drop(columns=[c for c in BLACKLIST if c in df.columns])
    print("Blacklisted and dropped: F3912, F2230")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    n_rows = len(df)

    # 2. Drop fully null columns (excluding preserved)
    null_counts = df.isnull().sum()
    all_null_cols = [c for c in df.columns[null_counts == n_rows] if c not in PRESERVE_COLS]
    df = df.drop(columns=all_null_cols)
    print(f"Dropped {len(all_null_cols)} fully null columns")

    # 3. Drop columns with >80% missing (excluding preserved)
    null_counts = df.isnull().sum()
    high_null_cols = [c for c in df.columns[null_counts / n_rows > 0.8] if c not in PRESERVE_COLS]
    df = df.drop(columns=high_null_cols)
    print(f"Dropped {len(high_null_cols)} columns with >80% missing values")

    # 4. Drop zero variance numeric columns (excluding preserved)
    numeric_cols = df.select_dtypes(include="number").columns
    zero_var_cols = [c for c in numeric_cols if c not in PRESERVE_COLS and df[c].std() == 0]
    df = df.drop(columns=zero_var_cols)
    print(f"Dropped {len(zero_var_cols)} zero variance columns")

    print(f"Final shape after cleaning: {df.shape}")

    log = {
        "dropped_all_null": all_null_cols,
        "dropped_high_null": high_null_cols,
        "dropped_zero_variance": zero_var_cols,
        "final_shape": df.shape,
    }

    return df, log


if __name__ == "__main__":
    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, cleaning_log = clean_dataset(raw)
    print(cleaning_log)
