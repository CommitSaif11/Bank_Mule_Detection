import os

import joblib
import pandas as pd
from sklearn.cluster import KMeans

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features

os.makedirs("saved_models", exist_ok=True)

CLUSTER_FEATURES = [
    "F115", "F670", "F1692", "F3887", "F3894",
    "F321", "F527", "F2082", "F2122", "F3891",
]


if __name__ == "__main__":
    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)

    y_full = engineered_df["F3924"]
    X_cluster = engineered_df[CLUSTER_FEATURES].copy()
    X_cluster = X_cluster.fillna(X_cluster.median())

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_cluster)
    clusters = pd.Series(clusters, index=X_cluster.index)

    print("\n" + "=" * 60)
    print("Cluster analysis (all 9082 accounts, raw features)")
    print("=" * 60)

    cluster_fraud_pct = {}
    for cluster_id in range(4):
        mask = clusters == cluster_id
        cluster_size = mask.sum()
        n_fraud = y_full[mask].sum()
        fraud_pct = 100 * n_fraud / cluster_size
        cluster_fraud_pct[cluster_id] = fraud_pct

        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster_size}")
        print(f"  Fraud count: {n_fraud} ({fraud_pct:.2f}%)")
        print("  Mean values of features:")
        for feat in CLUSTER_FEATURES:
            print(f"    {feat}: {X_cluster.loc[mask, feat].mean():.4f}")
        mode_occupation = X_cluster.loc[mask, "F3891"].mode().iloc[0]
        print(f"  Most common F3891 (occupation code): {mode_occupation}")

    # Assign typology labels based on fraud concentration
    ordered_clusters = sorted(cluster_fraud_pct, key=cluster_fraud_pct.get, reverse=True)
    label_names = ["Complicit Mule", "Recruited Mule", "Exploited Mule", "Low Risk"]
    cluster_to_label = {cluster_id: label_names[rank] for rank, cluster_id in enumerate(ordered_clusters)}

    print("\n" + "=" * 60)
    print("Typology mapping")
    print("=" * 60)
    for cluster_id in range(4):
        print(f"  Cluster {cluster_id} (fraud {cluster_fraud_pct[cluster_id]:.2f}%) -> {cluster_to_label[cluster_id]}")

    typology_labels = pd.Series(
        [cluster_to_label[c] for c in clusters],
        index=X_cluster.index,
        name="typology",
    )

    joblib.dump(kmeans, "saved_models/kmeans_typology.pkl")
    joblib.dump(typology_labels, "saved_models/typology_labels.pkl")

    print("\n" + "=" * 60)
    print("Fraud accounts by typology label")
    print("=" * 60)
    fraud_mask = y_full == 1
    print(typology_labels[fraud_mask].value_counts())

    print("\nSaved saved_models/kmeans_typology.pkl")
    print("Saved saved_models/typology_labels.pkl")
