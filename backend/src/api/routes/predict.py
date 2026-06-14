import joblib
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.utils import clean_record, clean_records

router = APIRouter()

_risk_df = pd.read_csv("saved_models/risk_scores.csv")
_typology_labels = joblib.load("saved_models/typology_labels.pkl")

VALID_TIERS = set(_risk_df["risk_tier"].unique())
VALID_TYPOLOGIES = set(_risk_df["typology_label"].unique())


def reload_state():
    global _risk_df, _typology_labels, VALID_TIERS, VALID_TYPOLOGIES
    _risk_df = pd.read_csv("saved_models/risk_scores.csv")
    _typology_labels = joblib.load("saved_models/typology_labels.pkl")
    VALID_TIERS = set(_risk_df["risk_tier"].unique())
    VALID_TYPOLOGIES = set(_risk_df["typology_label"].unique())


@router.get("/accounts")
def get_accounts(tier: str = None, typology: str = None):
    try:
        df = _risk_df

        if tier is not None:
            if tier not in VALID_TIERS:
                raise HTTPException(status_code=400, detail=f"Invalid tier '{tier}'. Valid values: {sorted(VALID_TIERS)}")
            df = df[df["risk_tier"] == tier]

        if typology is not None:
            if typology not in VALID_TYPOLOGIES:
                raise HTTPException(status_code=400, detail=f"Invalid typology '{typology}'. Valid values: {sorted(VALID_TYPOLOGIES)}")
            df = df[df["typology_label"] == typology]

        cols = ["account_index", "risk_score", "risk_tier", "typology_label", "ml_score", "true_label"]
        return clean_records(df[cols].to_dict(orient="records"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch accounts: {e}")


@router.get("/accounts/{account_id}")
def get_account(account_id: int):
    try:
        row = _risk_df[_risk_df["account_index"] == account_id]
        if row.empty:
            raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
        return clean_record(row.iloc[0].to_dict())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch account {account_id}: {e}")


@router.get("/stats")
def get_stats():
    try:
        tier_counts = _risk_df["risk_tier"].value_counts()
        fraud = _risk_df[_risk_df["true_label"] == 1]
        legit = _risk_df[_risk_df["true_label"] == 0]

        return clean_record({
            "total_accounts": len(_risk_df),
            "critical": int(tier_counts.get("Critical", 0)),
            "high": int(tier_counts.get("High", 0)),
            "medium": int(tier_counts.get("Medium", 0)),
            "low": int(tier_counts.get("Low", 0)),
            "fraud_detected": int(len(fraud)),
            "mean_risk_score_fraud": float(fraud["risk_score"].mean()),
            "mean_risk_score_legit": float(legit["risk_score"].mean()),
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute stats: {e}")
