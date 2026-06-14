# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MuleNet — a fraud intelligence dashboard for the mule account detection system. Vite + React (v19) SPA styled with Tailwind CSS, served at `localhost:5173`, talking to the FastAPI backend (`../backend`) at `localhost:8000`.

## Commands

- `npm run dev` — start Vite dev server
- `npm run build` — production build
- `npm run lint` — run ESLint
- `npm run preview` — preview production build

## Architecture

- **`src/api/client.js`** — single axios instance (`baseURL: http://localhost:8000`) with one exported function per backend endpoint: `getStats`, `getAlerts`, `getAccounts`, `getAccount`, `getExplain`. All API calls go through here.
- **`src/context/ThemeContext.jsx`** — `ThemeProvider`/`useTheme` for light/dark mode. Persists to `localStorage` (`mulenet-theme`), falls back to `prefers-color-scheme`, toggles the `dark` class on `<html>` (Tailwind `darkMode: 'class'`).
- **`src/App.jsx`** — route table: `/` (Dashboard), `/alerts` (Alerts), `/accounts` (Accounts), `/account/:id` (Account). Fixed-width dark sidebar (`Navbar`, 240px) + flex-1 main content.
- **`src/pages/`** — one component per route. Each page independently fetches its own data in `useEffect`, with `Loading`/`ErrorMessage` components for loading and failure states.
- **`src/components/`** — shared UI: `RiskBadge` (tier pill), `RiskGauge` (animated SVG risk score gauge), `Navbar`, `Loading`, `ErrorMessage`.

## Data model conventions

Field names mirror the backend `risk_scores.csv` / explain response exactly — don't rename when consuming the API:
- `account_index`, `risk_score` (0-100), `risk_tier` (`Low`/`Medium`/`High`/`Critical`), `typology_label` / `typology` (`Complicit Mule`/`Recruited Mule`/`Exploited Mule`/`Low Risk`), `true_label` (0/1), `ml_score` (0-100 from `/api/alerts` and `/api/accounts`, but `ml_fraud_probability` from `/api/explain` is 0-1 — these are NOT the same scale).
- Risk tier colors are defined as Tailwind theme colors: `critical` (#ef4444), `high` (#f97316), `medium` (#eab308), `low` (#22c55e).
- `/api/explain/:id` returns `top_risk_factors` (top 5 SHAP features with `feature`, `shap_value`, `account_value`, `direction`) and `investigation_summary`.

## Conventions

- Tailwind v3 (not v4) — `tailwind.config.js` uses the classic `content`/`theme.extend`/`plugins` format with `darkMode: 'class'`.
- Every page must handle three states: loading (`<Loading />`), error (`<ErrorMessage />`), and loaded — never render before data arrives.
- When adding dark mode classes, always pair color utilities (`text-gray-900` → `dark:text-white`, `bg-white` → `dark:bg-gray-800`, etc.) — don't leave light-only backgrounds on elements that also need a dark surface.
