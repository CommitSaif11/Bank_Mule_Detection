import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features

base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
raw = pd.read_csv(os.path.join(base, "data/DataSet.csv"))
cleaned_df, _ = clean_dataset(raw)
engineered_df = engineer_features(cleaned_df)

X = engineered_df.drop(columns=["F3924"])
y = engineered_df["F3924"]

feature_names = joblib.load(os.path.join(base, "models/selected_features.pkl"))
feature_names = [c for c in feature_names if c in X.columns]
X = X[feature_names]

# 2. Train one LightGBM on full data
model = LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1)
model.fit(X, y)

# need an imputed version for sklearn DecisionTree later
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# 3. Top 20 by split importance
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
top20 = importances.head(20)

print("=" * 70)
print("Top 20 features by LightGBM split importance:")
for col, imp in top20.items():
    corr = X[col].corr(y) if pd.api.types.is_numeric_dtype(X[col]) else float("nan")
    fraud_mean = X[col][y == 1].mean()
    legit_mean = X[col][y == 0].mean()
    ratio = fraud_mean / legit_mean if legit_mean != 0 else float("inf")
    nunique = engineered_df[col].nunique(dropna=True) if col in engineered_df.columns else X[col].nunique(dropna=True)

    print(f"\n--- {col} ---")
    print(f"  Split importance: {imp}")
    print(f"  Correlation with F3924: {corr:.4f}" if pd.notna(corr) else "  Correlation: N/A")
    print(f"  Mean (fraud=1): {fraud_mean}")
    print(f"  Mean (legit=0): {legit_mean}")
    print(f"  Fraud/Legit ratio: {ratio}")
    print(f"  nunique: {nunique}")

# 4. Depth-1 decision tree for top 5 features
print("\n" + "=" * 70)
print("Depth-1 Decision Tree single-feature split test (top 5 features):")
top5 = top20.head(5).index.tolist()
leaky_found = []

for col in top5:
    dt = DecisionTreeClassifier(max_depth=1, random_state=42)
    dt.fit(X_imp[[col]], y)
    y_pred = dt.predict(X_imp[[col]])

    threshold = dt.tree_.threshold[0]
    accuracy = (y_pred == y).mean()

    from sklearn.metrics import recall_score, confusion_matrix
    recall = recall_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"\n--- {col} ---")
    print(f"  Split threshold: {threshold}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Recall (fraud): {recall:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    left_mask = X_imp[col] <= threshold
    right_mask = ~left_mask
    print(f"  Left side (<= {threshold}): fraud={y[left_mask].sum()}, legit={(y[left_mask]==0).sum()}")
    print(f"  Right side (> {threshold}): fraud={y[right_mask].sum()}, legit={(y[right_mask]==0).sum()}")

    if accuracy > 0.95 or recall > 0.90:
        leaky_found.append((col, accuracy, recall))

print("\n" + "=" * 70)
print("LEAKY FEATURES (depth-1 tree accuracy > 95% or recall > 90%):")
for col, acc, rec in leaky_found:
    print(f"  {col}: accuracy={acc:.4f}, recall={rec:.4f}")
if not leaky_found:
    print("  None found")
