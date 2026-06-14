import joblib
import pandas as pd

from src.preprocessing.clean import clean_dataset
from src.features.engineer import engineer_features

if __name__ == "__main__":
    raw = pd.read_csv("data/DataSet.csv")
    cleaned_df, _ = clean_dataset(raw)
    engineered_df = engineer_features(cleaned_df)

    fraud = engineered_df[engineered_df["F3924"] == 1].drop(columns=["F3924"])
    legit = engineered_df[engineered_df["F3924"] == 0].drop(columns=["F3924"])

    fraud_pct = fraud.quantile(0.90)
    legit_pct = legit.quantile(0.10)

    joblib.dump(fraud_pct, "saved_models/fraud_percentiles.pkl")
    joblib.dump(legit_pct, "saved_models/legit_percentiles.pkl")

    print("Saved saved_models/fraud_percentiles.pkl and legit_percentiles.pkl")
