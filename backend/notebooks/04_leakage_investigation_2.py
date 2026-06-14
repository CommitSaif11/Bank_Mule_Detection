import joblib
import pandas as pd

df = pd.read_csv("../data/DataSet.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])
if "F3912" in df.columns:
    df = df.drop(columns=["F3912"])

target = df["F3924"]
numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != "F3924"]

corrs = {}
for col in numeric_cols:
    s = df[col]
    if s.notna().sum() < 2 or s.nunique(dropna=True) < 2:
        continue
    corr = s.corr(target)
    if pd.notna(corr):
        corrs[col] = corr

corr_series = pd.Series(corrs)

print("=" * 60)
print("Columns with |correlation| > 0.3 with F3924:")
high_corr = corr_series[corr_series.abs() > 0.3].sort_values(key=abs, ascending=False)
for col, val in high_corr.items():
    print(f"  {col}: {val:.4f}")

# 3. Top 20 features from LightGBM model
model = joblib.load("../models/lgbm_model.pkl")
feature_names = joblib.load("../models/selected_features.pkl")

importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
top20 = importances.head(20)

print("\n" + "=" * 60)
print("Top 20 features by LightGBM importance:")
for col, imp in top20.items():
    print(f"  {col}: {imp}")

# 4. Details for top 20
print("\n" + "=" * 60)
print("Details for top 20 features:")
blacklist = set()
for col in top20.index:
    if col not in df.columns:
        print(f"\n--- {col} ---")
        print("  Not in raw dataframe (engineered feature) - skipping detail")
        continue
    s = df[col]
    corr = corr_series.get(col, float("nan"))
    if pd.api.types.is_numeric_dtype(s):
        fraud_mean = s[target == 1].mean()
        legit_mean = s[target == 0].mean()
    else:
        fraud_mean = float("nan")
        legit_mean = float("nan")
    nunique = s.nunique(dropna=True)

    print(f"\n--- {col} ---")
    print(f"  Correlation with F3924: {corr:.4f}" if pd.notna(corr) else "  Correlation with F3924: N/A")
    print(f"  Mean (fraud=1): {fraud_mean}")
    print(f"  Mean (legit=0): {legit_mean}")
    print(f"  nunique: {nunique}")

    # 5. Flag checks
    flag = False
    reason = []
    if pd.notna(corr) and abs(corr) > 0.3:
        flag = True
        reason.append("corr>0.3")
    if pd.notna(legit_mean) and legit_mean != 0 and pd.notna(fraud_mean):
        ratio = fraud_mean / legit_mean
        if ratio > 5 or (ratio < 0.2 and ratio >= 0):
            flag = True
            reason.append(f"ratio={ratio:.4f}")
    if flag:
        blacklist.add(col)
        print(f"  FLAGGED: {', '.join(reason)}")

# also add all high-corr columns from step 2
for col in high_corr.index:
    blacklist.add(col)

print("\n" + "=" * 60)
print("FINAL BLACKLIST of suspected leakage columns:")
for col in sorted(blacklist):
    print(f"  {col}")
