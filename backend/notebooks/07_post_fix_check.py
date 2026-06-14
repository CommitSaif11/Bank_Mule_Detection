import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.tree import DecisionTreeClassifier

base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model = joblib.load(os.path.join(base, "models/lgbm_model.pkl"))
feature_names = joblib.load(os.path.join(base, "models/selected_features.pkl"))

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features

raw = pd.read_csv(os.path.join(base, "data/DataSet.csv"))
cleaned_df, _ = clean_dataset(raw)
engineered_df = engineer_features(cleaned_df)

X = engineered_df[feature_names]
y = engineered_df["F3924"]

imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
top10 = importances.head(10)

print("=" * 70)
print("Top 10 features by LightGBM importance (post-fix model):")
flagged = []
for col, imp in top10.items():
    corr = X[col].corr(y)
    fraud_mean = X[col][y == 1].mean()
    legit_mean = X[col][y == 0].mean()
    ratio = fraud_mean / legit_mean if legit_mean != 0 else float("inf")

    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt.fit(X_imp[[col]], y)
    y_pred = dt.predict(X_imp[[col]])
    recall = recall_score(y, y_pred)

    print(f"\n--- {col} ---")
    print(f"  Importance: {imp}")
    print(f"  Correlation with F3924: {corr:.4f}")
    print(f"  Mean (fraud=1): {fraud_mean}")
    print(f"  Mean (legit=0): {legit_mean}")
    print(f"  Fraud/Legit ratio: {ratio}")
    print(f"  Depth-1 tree recall: {recall:.4f}")

    if abs(corr) > 0.3 or recall > 0.9:
        flagged.append((col, corr, recall))

print("\n" + "=" * 70)
print("FLAGGED (correlation > 0.3 or depth-1 recall > 0.9):")
if flagged:
    for col, corr, rec in flagged:
        print(f"  {col}: corr={corr:.4f}, depth1_recall={rec:.4f}")
else:
    print("  None - no remaining leakage detected")
