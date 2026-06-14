import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("../data/DataSet.csv")

os.makedirs("../reports", exist_ok=True)

bank_cols = ["F115", "F321", "F527", "F531", "F670", "F1692", "F2082", "F2122",
              "F2582", "F2678", "F2737", "F2956", "F3043", "F3836", "F3887",
              "F3889", "F3891", "F3894"]

dist_cols = ["F115", "F670", "F1692", "F3887", "F3894"]

saved_files = []

# 1. Class distribution bar chart
plt.figure(figsize=(6, 5))
ax = sns.countplot(x="F3924", data=df)
ax.set_title("Class Distribution (F3924)")
ax.set_xlabel("Class (0 = Legit, 1 = Fraud)")
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha="center", va="bottom")
plt.tight_layout()
path = "../reports/class_distribution.png"
plt.savefig(path, dpi=150)
plt.close()
saved_files.append(path)

# 2. Feature distributions: fraud vs legit
fig, axes = plt.subplots(1, len(dist_cols), figsize=(5 * len(dist_cols), 4))
for ax, col in zip(axes, dist_cols):
    sns.kdeplot(data=df[df["F3924"] == 0], x=col, ax=ax, label="Legit", fill=True, common_norm=False)
    sns.kdeplot(data=df[df["F3924"] == 1], x=col, ax=ax, label="Fraud", fill=True, common_norm=False)
    ax.set_title(f"{col} Distribution")
    ax.set_xlabel(col)
    ax.set_ylabel("Density")
    ax.legend()
plt.tight_layout()
path = "../reports/feature_distributions.png"
plt.savefig(path, dpi=150)
plt.close()
saved_files.append(path)

# 3. Missing data percentage heatmap for the 18 bank-listed features
missing_pct = df[bank_cols].isnull().mean() * 100
plt.figure(figsize=(8, 6))
sns.heatmap(missing_pct.to_frame(name="Missing %"), annot=True, fmt=".1f", cmap="Reds", cbar=True)
plt.title("Missing Data Percentage - Bank-Listed Features")
plt.xlabel("")
plt.ylabel("Feature")
plt.tight_layout()
path = "../reports/missing_heatmap.png"
plt.savefig(path, dpi=150)
plt.close()
saved_files.append(path)

# 4. Correlation matrix for numeric subset of those 18 features
numeric_bank_cols = [c for c in bank_cols if pd.api.types.is_numeric_dtype(df[c])]
corr = df[numeric_bank_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation Matrix - Numeric Bank-Listed Features")
plt.tight_layout()
path = "../reports/correlation_matrix.png"
plt.savefig(path, dpi=150)
plt.close()
saved_files.append(path)

# 5. F3891 (occupation) vs fraud rate
fraud_rate = df.groupby("F3891")["F3924"].mean().sort_values(ascending=False) * 100
plt.figure(figsize=(8, 5))
ax = sns.barplot(x=fraud_rate.index, y=fraud_rate.values)
ax.set_title("Fraud Rate by Occupation (F3891)")
ax.set_xlabel("Occupation")
ax.set_ylabel("Fraud Rate (%)")
plt.xticks(rotation=45, ha="right")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                 ha="center", va="bottom")
plt.tight_layout()
path = "../reports/occupation_fraud_rate.png"
plt.savefig(path, dpi=150)
plt.close()
saved_files.append(path)

print("All plots saved successfully")
for f in saved_files:
    print(f)
