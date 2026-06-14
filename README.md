# MuleNet — Fraud Intelligence Platform

AI/ML-powered mule account detection system for banks.
Detects suspicious accounts, assigns risk scores, and
generates explainable investigation reports.

## Project Structure

mule_account_detection/
├── backend/          FastAPI + ML pipeline
└── frontend/         React + Tailwind dashboard

## Prerequisites

- Python 3.10+
- Node.js 18+
- Git

## Setup Instructions

### Step 1 — Clone the repository
git clone <your-repo-url>
cd mule_account_detection

### Step 2 — Backend setup
cd backend
python -m venv .venv

Windows:
.venv\Scripts\activate

Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt

### Step 3 — Add the dataset
Place DataSet.csv inside backend/data/
(Dataset is not included in the repo due to size)

### Step 4 — Run the ML pipeline
This generates all model files needed to run the API.
python -m src.pipeline.run_pipeline

This will take approximately 2-3 minutes.
It creates all .pkl files in backend/models/ and
backend/saved_models/

### Step 5 — Start the backend
uvicorn src.api.main:app --reload --port 8000

API will be available at http://localhost:8000
API docs at http://localhost:8000/docs

### Step 6 — Frontend setup (new terminal)
cd frontend
npm install
npm run dev

Frontend will be available at http://localhost:5173

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/stats | Dashboard statistics |
| GET | /api/alerts | High/Critical risk accounts |
| GET | /api/accounts | All accounts (filterable) |
| GET | /api/accounts/{id} | Single account profile |
| GET | /api/explain/{id} | SHAP investigation report |
| POST | /api/upload | Upload new CSV dataset |
| POST | /api/score/live | Score single account instantly |
| GET | /api/health | API health check |

## Model Performance

| Metric | Score |
|--------|-------|
| CV Recall (fraud) | 72.8% ± 8.6% |
| CV Precision (fraud) | 98.7% ± 2.7% |
| AUC-ROC | 99.1% ± 0.5% |
| AUC-PR | 92.9% ± 4.2% |
| High/Critical detection | 96.3% (78/81) |

Evaluated via 5-fold stratified cross-validation.
No synthetic data in test sets.

## Tech Stack

Backend: FastAPI, LightGBM, XGBoost, SHAP, scikit-learn,
         imbalanced-learn, pandas, numpy

Frontend: React, Vite, Tailwind CSS, Recharts,
          React Router, Axios, lucide-react

## Key Features

- Batch scoring of entire account portfolios
- Live single-account scoring API (<50ms)
- SHAP explainability for every flagged account
- Mule typology classification (4 types)
- Risk Fusion Engine (0-100 score)
- CSV upload with full pipeline rerun
- Dark mode dashboard

## Deployment

See Render deployment guide below.

Backend: Deploy as Web Service (Python)
Start command: uvicorn src.api.main:app --host 0.0.0.0 --port $PORT

Frontend: Deploy as Static Site
Build command: npm run build
Publish directory: dist
