from fastapi import APIRouter, HTTPException

from src.scoring.explainer import generate_report
from src.api.utils import clean_record

router = APIRouter()


@router.get("/explain/{account_id}")
def explain_account(account_id: int):
    try:
        report = generate_report(account_id)
        return clean_record(report)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate explanation for account {account_id}: {e}")
