from fastapi import APIRouter, HTTPException

from src.api.utils import clean_record, clean_records
from src.api.dependencies import get_risk_scores, get_typology_labels

router = APIRouter()


def _not_ready(e):
    raise HTTPException(status_code=503, detail="Models not yet loaded. Please upload a dataset first.")


@router.get("/accounts")
def get_accounts(tier: str = None, typology: str = None):
    try:
        df = get_risk_scores()
    except FileNotFoundError as e:
        _not_ready(e)

    try:
        valid_tiers = set(df["risk_tier"].unique())
        valid_typologies = set(df["typology_label"].unique())

        if tier is not None:
            if tier not in valid_tiers:
                raise HTTPException(status_code=400, detail=f"Invalid tier '{tier}'. Valid values: {sorted(valid_tiers)}")
            df = df[df["risk_tier"] == tier]

        if typology is not None:
            if typology not in valid_typologies:
                raise HTTPException(status_code=400, detail=f"Invalid typology '{typology}'. Valid values: {sorted(valid_typologies)}")
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
        df = get_risk_scores()
    except FileNotFoundError as e:
        _not_ready(e)

    try:
        row = df[df["account_index"] == account_id]
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
        df = get_risk_scores()
    except FileNotFoundError as e:
        _not_ready(e)

    try:
        tier_counts = df["risk_tier"].value_counts()
        fraud = df[df["true_label"] == 1]
        legit = df[df["true_label"] == 0]

        return clean_record({
            "total_accounts": len(df),
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
