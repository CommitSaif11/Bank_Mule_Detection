import os
from datetime import datetime, timezone

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from src.pipeline.run_pipeline import run_full_pipeline, PipelineStepError
from src.api.dependencies import clear_cache

router = APIRouter()

UPLOAD_PATH = "data/uploaded_dataset.csv"

_status = {
    "current_dataset": "DataSet.csv",
    "last_updated": None,
    "total_accounts": 0,
}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a CSV")

    try:
        contents = await file.read()
        os.makedirs("data", exist_ok=True)
        with open(UPLOAD_PATH, "wb") as f:
            f.write(contents)
    except Exception as e:
        return {"status": "error", "step": "save_upload", "message": str(e)}

    try:
        raw_df = pd.read_csv(UPLOAD_PATH)
    except Exception as e:
        return {"status": "error", "step": "read_csv", "message": str(e)}

    try:
        result = run_full_pipeline(raw_df)
    except PipelineStepError as e:
        return {"status": "error", "step": e.step, "message": e.message}
    except Exception as e:
        return {"status": "error", "step": "pipeline", "message": str(e)}

    try:
        clear_cache()
    except Exception as e:
        return {"status": "error", "step": "reload_state", "message": str(e)}

    results = result["results"]
    tier_counts = results["risk_tier"].value_counts()

    stats = {
        "total_accounts": len(results),
        "critical": int(tier_counts.get("Critical", 0)),
        "high": int(tier_counts.get("High", 0)),
        "medium": int(tier_counts.get("Medium", 0)),
        "low": int(tier_counts.get("Low", 0)),
        "columns_after_cleaning": result["clean_log"]["final_shape"][1],
        "features_selected": len(result["feature_names"]),
    }
    if result["has_target"]:
        stats["fraud_detected"] = int((results["true_label"] == 1).sum())

    _status["current_dataset"] = "uploaded_dataset.csv"
    _status["last_updated"] = datetime.now(timezone.utc).isoformat()
    _status["total_accounts"] = len(results)

    return {
        "status": "success",
        "message": "Pipeline completed successfully",
        "stats": stats,
    }


@router.get("/upload/status")
def upload_status():
    return _status
