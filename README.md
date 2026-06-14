# MuleNet — Fraud Intelligence Platform

AI/ML-powered mule account detection system for banks.
Detects suspicious accounts, assigns risk scores, and
generates explainable investigation reports.

## Project Structure

```
mule_account_detection/
├── backend/          FastAPI + ML pipeline
└── frontend/         React + Tailwind dashboard
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Git**
- **uv** (fast Python package/environment manager — instructions below)

This guide assumes **no prior setup** — every command can be copy-pasted as-is.

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/CommitSaif11/Bank_Mule_Detection.git
cd Bank_Mule_Detection
```

---

## Step 2 — Install `uv`

`uv` is a fast Python package and virtual environment manager (replacement for `pip` + `venv`). We use it to create the backend environment and install dependencies.

Check if `uv` is already installed:

```bash
uv --version
```

If not found, install it:

- **Windows (PowerShell)**:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```
- **Mac / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

After installation, **close and reopen your terminal**, then verify:

```bash
uv --version
```

> **Note on Python versions**: You do **not** need to manually install Python 3.10. `uv` can download and manage Python versions for you automatically — when you run `uv venv --python 3.10 .venv` in Step 3, `uv` will fetch Python 3.10 itself if it isn't already available on your system.

---

## Step 3 — Backend setup

Navigate into the `backend/` folder:

```bash
cd backend
```

### 4.1 — Create a virtual environment (Python 3.10) using `uv`

```bash
uv venv --python 3.10 .venv
```

This creates a `.venv/` folder inside `backend/` containing an isolated Python 3.10 environment.

### 4.2 — Activate the virtual environment

- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\activate
  ```
- **Mac / Linux**:
  ```bash
  source .venv/bin/activate
  ```

You should now see `(.venv)` at the start of your terminal prompt.

### 4.3 — Install all dependencies from `requirements.txt`

Using `uv` (recommended — much faster than plain `pip`):

```bash
uv pip install -r requirements.txt --python .venv/Scripts/python.exe
```

> On Mac/Linux, use the Unix path instead:
> ```bash
> uv pip install -r requirements.txt --python .venv/bin/python
> ```

This installs FastAPI, LightGBM, XGBoost, SHAP, scikit-learn, imbalanced-learn, pandas, numpy, and all other required packages.

---

## Step 4 — Add the dataset

The dataset is **not included** in this repository (too large for GitHub).

1. Obtain `DataSet.csv` from the project owner.
2. Place it inside the `backend/data/` folder so the path is:
   ```
   backend/data/DataSet.csv
   ```

---

## Step 5 — Run the ML pipeline

This step trains all models (LightGBM, XGBoost, Random Forest, Isolation Forest, KMeans) and generates every `.pkl` file the API needs. It must be run **once** before starting the backend (and again any time you upload a new dataset, if not done automatically via the Upload page).

Make sure you are inside `backend/` and your virtual environment is activated, then run:

```bash
python -m src.pipeline.run_pipeline
```

This will take approximately **2-3 minutes**. When it finishes, you will see new files created in:
- `backend/models/` (e.g. `lgbm_model.pkl`, `imputer.pkl`, `scaler.pkl`, `selected_features.pkl`)
- `backend/saved_models/` (e.g. `isolation_forest.pkl`, `kmeans_typology.pkl`, `risk_scores.csv`, `feature_medians.pkl`)

---

## Step 6 — Start the backend server

Still inside `backend/`, with the virtual environment activated, run:

```bash
uvicorn src.api.main:app --reload --port 8000
```

If that command is not found, run it through the venv's Python directly:

```bash
.venv/Scripts/python -m uvicorn src.api.main:app --reload --port 8000
```

Once running:
- API base URL: http://localhost:8000
- Interactive API docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/api/health

Keep this terminal window open — the backend must stay running.

---

## Step 7 — Frontend setup

Open a **new terminal window** (leave the backend running in the first one).

### 8.1 — Navigate to the frontend folder

```bash
cd frontend
```

### 8.2 — Install Node dependencies

```bash
npm install
```

### 8.3 — Start the frontend dev server

```bash
npm run dev
```

Once running, open your browser to:

```
http://localhost:5173
```

The dashboard will load and connect automatically to the backend at `http://localhost:8000` (configured via `frontend/.env.development`).

---

## Step 8 — Verify everything works

1. Open http://localhost:5173 — you should see the MuleNet dashboard with stats populated.
2. Open http://localhost:8000/docs — you should see the FastAPI Swagger documentation.
3. If the dashboard shows an error/empty state, confirm:
   - The backend terminal shows no errors and is still running.
   - Step 5 (pipeline run) completed successfully and created `.pkl` files.

---

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

Backend: Deploy as Web Service (Python)
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
```

Frontend: Deploy as Static Site
```bash
npm run build
```
Publish directory: `dist`

## Troubleshooting

- **`uv: command not found`** — restart your terminal after installing `uv`, or ensure `~/.local/bin` (Mac/Linux) or `%USERPROFILE%\.local\bin` (Windows) is in your `PATH`.
- **`uvicorn: command not found`** — use `.venv/Scripts/python -m uvicorn ...` (Windows) or `.venv/bin/python -m uvicorn ...` (Mac/Linux) instead.
- **Port 8000 already in use** — stop the existing process or run on a different port: `uvicorn src.api.main:app --reload --port 8001` (and update `frontend/.env.development` accordingly).
- **Frontend shows "backend unreachable"** — confirm the backend is running on port 8000 and `frontend/.env.development` has `VITE_API_URL=http://localhost:8000`.
