import os

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features
from src.preprocessing.split_and_impute import prepare_data
from src.features.select_features import select_features
from src.models.typology import CLUSTER_FEATURES
from src.scoring.risk_fusion import TYPOLOGY_BOOST, assign_tier


class PipelineStepError(Exception):
    def __init__(self, step, message):
        self.step = step
        self.message = message
        super().__init__(f"{step}: {message}")


def run_full_pipeline(raw_df: pd.DataFrame) -> dict:
    """Run the full clean -> engineer -> train -> score pipeline and persist all artifacts.

    Raises PipelineStepError(step, message) on failure, naming the step that failed.
    """
    os.makedirs("models", exist_ok=True)
    os.makedirs("saved_models", exist_ok=True)

    has_target = "F3924" in raw_df.columns
    ctx = {"step": "init"}

    try:
        return _run(raw_df, has_target, ctx)
    except PipelineStepError:
        raise
    except Exception as e:
        raise PipelineStepError(ctx["step"], str(e))


def _run(raw_df, has_target, ctx):
    df = raw_df.copy()
    if not has_target:
        df["F3924"] = 0

    ctx["step"] = "clean_dataset"
    cleaned_df, clean_log = clean_dataset(df)

    ctx["step"] = "engineer_features"
    engineered_df = engineer_features(cleaned_df)

    ctx["step"] = "prepare_data"
    X_train, X_test, y_train, y_test = prepare_data(engineered_df)

    ctx["step"] = "select_features"
    X_train_sel, X_test_sel, feature_names = select_features(X_train, y_train, X_test)

    ctx["step"] = "train_model"
    if has_target and y_train.nunique() > 1:
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_sel, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train_sel, y_train

    lgbm_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_model.fit(X_train_resampled, y_train_resampled)
    joblib.dump(lgbm_model, "models/lgbm_model.pkl")

    ctx["step"] = "feature_medians"
    feature_medians = engineered_df.loc[X_train.index, feature_names].median()
    joblib.dump(feature_medians, "saved_models/feature_medians.pkl")

    engineered_features_only = engineered_df.drop(columns=["F3924"])
    if has_target and (engineered_df["F3924"] == 1).any():
        fraud_percentiles = engineered_features_only[engineered_df["F3924"] == 1].quantile(0.90)
    else:
        fraud_percentiles = feature_medians.reindex(engineered_features_only.columns, fill_value=0)
    if has_target and (engineered_df["F3924"] == 0).any():
        legit_percentiles = engineered_features_only[engineered_df["F3924"] == 0].quantile(0.10)
    else:
        legit_percentiles = feature_medians.reindex(engineered_features_only.columns, fill_value=0)
    joblib.dump(fraud_percentiles, "saved_models/fraud_percentiles.pkl")
    joblib.dump(legit_percentiles, "saved_models/legit_percentiles.pkl")

    ctx["step"] = "build_full_matrix"
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")

    X_full = engineered_df.drop(columns=["F3924"])
    y_full = engineered_df["F3924"]

    X_full_imputed = imputer.transform(X_full)
    X_full_scaled = scaler.transform(X_full_imputed)
    X_full_scaled = pd.DataFrame(X_full_scaled, columns=X_full.columns, index=X_full.index)
    X_full_selected = X_full_scaled[feature_names]

    ctx["step"] = "isolation_forest"
    iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train_sel)
    raw_scores = iso_forest.decision_function(X_full_selected)
    min_score, max_score = raw_scores.min(), raw_scores.max()
    anomaly_scores = 100 * (max_score - raw_scores) / (max_score - min_score)
    anomaly_scores = pd.Series(anomaly_scores, index=X_full_selected.index, name="anomaly_score")

    joblib.dump(iso_forest, "saved_models/isolation_forest.pkl")
    joblib.dump(anomaly_scores, "saved_models/anomaly_scores.pkl")
    joblib.dump((float(min_score), float(max_score)), "saved_models/anomaly_score_range.pkl")

    ctx["step"] = "kmeans_typology"
    X_cluster = engineered_df[CLUSTER_FEATURES].copy()
    X_cluster = X_cluster.fillna(X_cluster.median())

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_cluster)
    clusters = pd.Series(clusters, index=X_cluster.index)

    cluster_fraud_pct = {}
    for cluster_id in range(4):
        mask = clusters == cluster_id
        cluster_size = mask.sum()
        if has_target and cluster_size > 0:
            cluster_fraud_pct[cluster_id] = 100 * y_full[mask].sum() / cluster_size
        else:
            cluster_fraud_pct[cluster_id] = 0

    ordered_clusters = sorted(cluster_fraud_pct, key=cluster_fraud_pct.get, reverse=True)
    label_names = ["Complicit Mule", "Recruited Mule", "Exploited Mule", "Low Risk"]
    cluster_to_label = {cluster_id: label_names[rank] for rank, cluster_id in enumerate(ordered_clusters)}

    typology_labels = pd.Series(
        [cluster_to_label[c] for c in clusters],
        index=X_cluster.index,
        name="typology",
    )

    joblib.dump(kmeans, "saved_models/kmeans_typology.pkl")
    joblib.dump(typology_labels, "saved_models/typology_labels.pkl")
    joblib.dump(cluster_to_label, "saved_models/typology_cluster_map.pkl")

    ctx["step"] = "risk_fusion"
    ml_proba = lgbm_model.predict_proba(X_full_selected)[:, 1]
    ml_score = pd.Series(ml_proba * 100, index=X_full_selected.index, name="ml_score")

    typology_boost = typology_labels.map(TYPOLOGY_BOOST)

    risk_score = (
        ml_score * 0.75
        + anomaly_scores * 0.15
        + typology_boost * 0.10 * (100 / 8)
    )
    risk_score = risk_score.clip(0, 100)
    risk_tier = risk_score.apply(assign_tier)

    true_label = y_full if has_target else pd.Series(-1, index=y_full.index)

    results = pd.DataFrame({
        "account_index": X_full_selected.index,
        "ml_score": ml_score,
        "anomaly_score": anomaly_scores,
        "typology_label": typology_labels,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "true_label": true_label,
    })

    ctx["step"] = "save_results"
    results.to_csv("saved_models/risk_scores.csv", index=False)

    return {
        "results": results,
        "clean_log": clean_log,
        "feature_names": feature_names,
        "has_target": has_target,
    }
