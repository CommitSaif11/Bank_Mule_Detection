from fastapi import APIRouter, HTTPException

from src.api.utils import clean_records
from src.api.dependencies import get_risk_scores

router = APIRouter()


@router.get("/alerts")
def get_alerts():
    try:
        df = get_risk_scores()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Models not yet loaded. Please upload a dataset first.")

    try:
        df = df[df["risk_tier"].isin(["Critical", "High"])]
        df = df.sort_values("risk_score", ascending=False)
        cols = ["account_index", "risk_score", "risk_tier", "typology_label", "ml_score"]
        return clean_records(df[cols].to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch alerts: {e}")


@router.get("/alerts/summary")
def get_alerts_summary():
    try:
        df = get_risk_scores()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Models not yet loaded. Please upload a dataset first.")

    try:
        df = df[df["risk_tier"].isin(["Critical", "High"])]
        return {
            "by_tier": df["risk_tier"].value_counts().to_dict(),
            "by_typology": df["typology_label"].value_counts().to_dict(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch alerts summary: {e}")
