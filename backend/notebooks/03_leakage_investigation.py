import numpy as np
import pandas as pd

df = pd.read_csv("../data/DataSet.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

target = df["F3924"]
numeric_cols = df.select_dtypes(include="number").columns
numeric_cols = [c for c in numeric_cols if c != "F3924"]

# 2. Point-biserial correlation = pearson correlation between binary target and numeric column
corrs = {}
for col in numeric_cols:
    s = df[col]
    if s.notna().sum() < 2 or s.nunique(dropna=True) < 2:
        continue
    corr = s.corr(target)
    if pd.notna(corr):
        corrs[col] = corr

corr_series = pd.Series(corrs)
top30 = corr_series.abs().sort_values(ascending=False).head(30)

print("=" * 60)
print("TOP 30 columns by absolute correlation with F3924:")
for col, abs_val in top30.items():
    print(f"  {col}: {corr_series[col]:.4f}")

# 3. Top 10 details
print("\n" + "=" * 60)
print("Top 10 most correlated columns - details:")
top10 = top30.head(10).index.tolist()
for col in top10:
    s = df[col]
    fraud_mean = s[target == 1].mean()
    legit_mean = s[target == 0].mean()
    nunique = s.nunique(dropna=True)
    nonnull_fraud = s[target == 1].notna().sum()
    nonnull_legit = s[target == 0].notna().sum()
    only_fraud = nonnull_legit == 0 and nonnull_fraud > 0
    only_legit = nonnull_fraud == 0 and nonnull_legit > 0

    print(f"\n--- {col} (corr={corr_series[col]:.4f}) ---")
    print(f"  Mean (fraud=1): {fraud_mean}")
    print(f"  Mean (legit=0): {legit_mean}")
    print(f"  nunique: {nunique}")
    print(f"  non-null count fraud: {nonnull_fraud}, non-null count legit: {nonnull_legit}")
    print(f"  Only non-null for fraud: {only_fraud}")
    print(f"  Only non-null for legit: {only_legit}")

# 4. Columns F3900 to F3924
print("\n" + "=" * 60)
print("Columns F3900 to F3924 - value counts and correlation with F3924:")
for i in range(3900, 3925):
    col = f"F{i}"
    if col not in df.columns:
        continue
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))
    if col != "F3924" and pd.api.types.is_numeric_dtype(df[col]):
        s = df[col]
        if s.notna().sum() >= 2 and s.nunique(dropna=True) >= 2:
            corr = s.corr(target)
            print(f"  Correlation with F3924: {corr:.4f}")
        else:
            print("  Correlation with F3924: N/A (constant or all null)")
    else:
        print("  Correlation with F3924: N/A (non-numeric or is target)")

# 5. Suspected leakage columns
print("\n" + "=" * 60)
suspected = corr_series[corr_series.abs() > 0.5].sort_values(key=abs, ascending=False)
print("SUSPECTED LEAKAGE COLUMNS (abs correlation > 0.5):")
for col, val in suspected.items():
    print(f"  {col}: {val:.4f}")
