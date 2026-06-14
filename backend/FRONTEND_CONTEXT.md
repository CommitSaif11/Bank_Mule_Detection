# MuleNet Frontend Context

## What this project is
MuleNet is a mule account detection platform. It scores 9,082 bank accounts 
for fraud risk using ML. It is a BATCH system — not real-time.

## Backend API (running on localhost:8000)
- GET /api/stats → total accounts, critical/high/medium/low counts, fraud detected
- GET /api/alerts → 78 accounts (Critical+High), sorted by risk score desc
- GET /api/accounts → all 9082 accounts, supports ?tier= and ?typology= filters
- GET /api/accounts/{id} → single account profile
- GET /api/explain/{id} → SHAP investigation report with top 5 risk factors

## Key numbers to display
- Total accounts: 9,082
- Critical alerts: 60
- High alerts: 18  
- Fraud detected: 81
- Mean risk score fraud: 81.7
- Mean risk score legit: 8.3
- Fraud accounts in High/Critical: 78 out of 81 (96.3%)

## Risk tiers
- Critical (81-100) → red
- High (61-80) → orange
- Medium (31-60) → yellow
- Low (0-30) → green

## Mule typologies
- Complicit Mule → knowingly involved, long-term
- Recruited Mule → paid commission, short-term
- Exploited Mule → account taken over
- Low Risk → normal behavior

## SHAP explain response shape
{
  account_index, risk_score, risk_tier, typology,
  ml_fraud_probability, true_label,
  top_risk_factors: [
    {feature, shap_value, account_value, direction}
  ],
  investigation_summary: "plain English string"
}

## Design requirements
- Modern, dark or light professional theme
- Judges will focus on this — make it impressive
- Every number from the backend must show on screen
- Dashboard, Alerts table, Account investigation page
- Risk score should look like a gauge or large prominent number
- SHAP factors should be visual bars not just text
- Mobile responsive
