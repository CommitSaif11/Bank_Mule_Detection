# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MuleNet — mule account fraud detection. A Python ML pipeline (`backend/`) trains a binary classifier (`F3924` target, ~0.9% fraud rate, 9,082 accounts) and serves risk scores + SHAP explanations via FastAPI. A Vite + React dashboard (`frontend/`) visualizes the results.

Two independent apps, each with its own CLAUDE.md — read the relevant one before working in that directory:
- **[backend/CLAUDE.md](backend/CLAUDE.md)** — ML pipeline (clean → engineer → split/impute → select features → train), known data-leakage issue, environment setup (`.venv`, `uv`).
- **[frontend/CLAUDE.md](frontend/CLAUDE.md)** — React dashboard architecture, API client conventions, dark mode.

## Running both together

Backend (from `backend/`, port 8000):
```
.venv/Scripts/python -m uvicorn src.api.main:app --port 8000 --reload
```

Frontend (from `frontend/`, port 5173):
```
npm run dev
```

The frontend's `src/api/client.js` expects the backend at `http://localhost:8000` (CORS is open to `*` for local dev). Both must be running for the dashboard to show real data — `ErrorMessage` is shown if the backend is unreachable.

## Backend API surface (consumed by frontend)

- `GET /api/health` — status check (models loaded, account count, timestamp)
- `GET /api/stats` — total/critical/high/medium/low counts, fraud detected, mean risk scores
- `GET /api/alerts` — Critical+High accounts sorted by risk score desc
- `GET /api/alerts/summary` — alert counts by tier/typology
- `GET /api/accounts?tier=&typology=` — filtered account list (400 on invalid filter value)
- `GET /api/accounts/{id}` — single account (404 if not found)
- `GET /api/explain/{id}` — SHAP report with top 5 risk factors + investigation summary (404 if not found)

All endpoints sanitize NaN/Infinity to `null` and return proper `HTTPException` errors (no raw 500s).

## Data scale gotcha

`ml_score` from `/api/alerts` and `/api/accounts` is on a **0-100** scale, but `ml_fraud_probability` from `/api/explain` is **0-1**. Don't mix them up when displaying percentages.

## Known issue

Backend has had multiple target-leakage columns discovered (`F3912`, `F2230`, blacklisted in `src/preprocessing/clean.py`). See backend/CLAUDE.md for the leakage-detection methodology before trusting any new high-importance feature.
