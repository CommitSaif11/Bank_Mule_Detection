import numpy as np
from fastapi import APIRouter, HTTPException

from src.api.utils import clean_record
from src.api.dependencies import get_explainer_state

router = APIRouter()


@router.get("/explain/{account_id}")
def explain_account(account_id: int):
    try:
        state = get_explainer_state()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Models not yet loaded. Please upload a dataset first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load explainer state: {e}")

    try:
        X_processed = state["X_processed"]
        shap_values = state["shap_values"]
        risk_scores = state["risk_scores"]
        typology_labels = state["typology_labels"]
        feature_names = state["feature_names"]

        if account_id not in X_processed.index:
            raise HTTPException(status_code=404, detail=f"Account {account_id} not found")

        row_pos = X_processed.index.get_loc(account_id)

        risk_row = risk_scores.loc[account_id]
        risk_score = float(risk_row["risk_score"])
        risk_tier = str(risk_row["risk_tier"])
        ml_proba = float(risk_row["ml_score"]) / 100
        true_label = int(risk_row["true_label"])
        typology = str(typology_labels.loc[account_id])

        sv = shap_values[row_pos]
        account_values = X_processed.iloc[row_pos]

        order = np.argsort(-np.abs(sv))[:5]
        top_risk_factors = []
        for i in order:
            feature = feature_names[i]
            shap_val = float(sv[i])
            top_risk_factors.append({
                "feature": feature,
                "shap_value": shap_val,
                "account_value": float(account_values.iloc[i]),
                "direction": "increases risk" if shap_val > 0 else "decreases risk",
            })

        drivers = []
        for f in top_risk_factors[:3]:
            if f["direction"] == "increases risk":
                drivers.append(f"{f['feature']} strongly increases risk")
            else:
                drivers.append(f"{f['feature']} decreases risk")

        investigation_summary = (
            f"This account shows {risk_tier.upper()} risk (score: {risk_score:.1f}). "
            f"Typology: {typology}. "
            f"Key drivers: {', '.join(drivers)}."
        )

        report = {
            "account_index": account_id,
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "typology": typology,
            "ml_fraud_probability": ml_proba,
            "true_label": true_label,
            "top_risk_factors": top_risk_factors,
            "investigation_summary": investigation_summary,
        }
        return clean_record(report)
    except HTTPException:
        raise
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation for account {account_id}: {e}")
