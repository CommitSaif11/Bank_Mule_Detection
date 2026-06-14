from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import predict, alerts, explain, upload, live
from src.api.startup import startup_check

app = FastAPI(title="MuleNet API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api")
app.include_router(alerts.router, prefix="/api")
app.include_router(explain.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(live.router, prefix="/api")


@app.on_event("startup")
async def on_startup():
    try:
        startup_check()
    except Exception as e:
        print(f"Startup warning: {e}")
        print("API starting without pre-loaded models.")
        print("Models will load on first request.")


@app.get("/")
def root():
    return {"status": "MuleNet API running"}


@app.get("/api/health")
def health():
    try:
        risk_df = pd.read_csv("saved_models/risk_scores.csv")
        models_loaded = True
        total_accounts = len(risk_df)
    except Exception:
        models_loaded = False
        total_accounts = 0

    return {
        "status": "healthy",
        "models_loaded": models_loaded,
        "total_accounts": total_accounts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
