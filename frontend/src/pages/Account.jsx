import { useEffect, useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, FileText } from 'lucide-react'
import { getAccount, getExplain } from '../api/client'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import RiskGauge from '../components/RiskGauge'
import RiskBadge from '../components/RiskBadge'

const TYPOLOGY_DESCRIPTIONS = {
  'Complicit Mule': 'Knowingly involved in fraud network',
  'Recruited Mule': 'Paid commission for account access',
  'Exploited Mule': 'Account taken over without knowledge',
  'Low Risk': 'Normal behavioral profile',
}

const ACTION_BY_TIER = {
  Critical: 'Escalate immediately to fraud investigation team',
  High: 'Schedule review within 24 hours',
  Medium: 'Monitor and review within 7 days',
  Low: 'No immediate action required',
}

export default function Account() {
  const { id } = useParams()
  const [account, setAccount] = useState(null)
  const [explain, setExplain] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    Promise.all([getAccount(id), getExplain(id)])
      .then(([accRes, expRes]) => {
        setAccount(accRes.data)
        setExplain(expRes.data)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [id])

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} />

  const maxAbsShap = Math.max(...explain.top_risk_factors.map((f) => Math.abs(f.shap_value)), 0.001)

  return (
    <div className="space-y-6">
      <div>
        <Link to="/alerts" className="inline-flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-800 mb-3">
          <ArrowLeft size={16} />
          Back to Alerts
        </Link>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Account Investigation</h1>
        <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">Account ID: {explain.account_index}</p>
      </div>

      {/* Top section */}
      <div className="grid grid-cols-3 gap-5">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 flex flex-col items-center justify-center">
          <RiskGauge score={explain.risk_score} tier={explain.risk_tier} size={220} />
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-4">
            ML Fraud Probability: <span className="font-semibold text-gray-900 dark:text-white">{(explain.ml_fraud_probability * 100).toFixed(1)}%</span>
          </p>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4">Account Profile</h2>
          <div className="space-y-4 text-sm">
            <div>
              <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Risk Tier</p>
              <RiskBadge tier={explain.risk_tier} />
            </div>
            <div>
              <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Typology</p>
              <p className="font-semibold text-gray-900 dark:text-white">{explain.typology}</p>
              <p className="text-gray-500 dark:text-gray-400 text-xs mt-0.5">{TYPOLOGY_DESCRIPTIONS[explain.typology]}</p>
            </div>
            <div>
              <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">True Label</p>
              {explain.true_label === 1 ? (
                <span className="inline-block px-2.5 py-1 rounded-full text-xs font-semibold bg-critical text-white">Fraud</span>
              ) : (
                <span className="inline-block px-2.5 py-1 rounded-full text-xs font-semibold bg-low text-white">Legitimate</span>
              )}
            </div>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4">Risk Score Breakdown</h2>
          <div className="space-y-5">
            <ScoreBar label="ML Score" sublabel="dominant signal" value={explain.ml_fraud_probability * 100} color="#3b82f6" />
            {account?.anomaly_score != null && (
              <ScoreBar label="Anomaly Score" value={account.anomaly_score} color="#a855f7" />
            )}
            <ScoreBar label="Final Risk Score" value={explain.risk_score} color="#ef4444" bold />
          </div>
        </div>
      </div>

      {/* SHAP factors */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <h2 className="text-base font-semibold text-gray-900 dark:text-white">SHAP Feature Importance — Why this account was flagged</h2>
        <p className="text-gray-500 dark:text-gray-400 text-sm mb-4">Top 5 features driving the risk score</p>

        <div className="space-y-4">
          {explain.top_risk_factors.map((f) => {
            const widthPct = (Math.abs(f.shap_value) / maxAbsShap) * 50
            const increases = f.direction === 'increases risk'
            return (
              <div key={f.feature} className="border-b border-gray-50 dark:border-gray-700/50 pb-3 last:border-0 last:pb-0">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="font-medium text-gray-900 dark:text-white text-sm">{f.feature}</span>
                  <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${increases ? 'bg-red-50 text-critical' : 'bg-green-50 text-low'}`}>
                    {increases ? 'Increases Risk' : 'Decreases Risk'}
                  </span>
                </div>
                <div className="relative w-full h-3 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden flex">
                  <div className="w-1/2 flex justify-end">
                    {!increases && (
                      <div className="h-full bg-low rounded-l-full" style={{ width: `${widthPct * 2}%` }} />
                    )}
                  </div>
                  <div className="w-1/2 flex justify-start">
                    {increases && (
                      <div className="h-full bg-critical rounded-r-full" style={{ width: `${widthPct * 2}%` }} />
                    )}
                  </div>
                  <div className="absolute left-1/2 top-0 h-full w-px bg-gray-300 dark:bg-gray-600" />
                </div>
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span>Value: {f.account_value.toFixed(3)}</span>
                  <span>Impact: {f.shap_value >= 0 ? '+' : ''}{f.shap_value.toFixed(3)}</span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Investigation summary */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div className="flex items-start gap-3">
          <div className="flex items-center justify-center w-10 h-10 rounded-lg bg-blue-50 text-blue-500 flex-shrink-0">
            <FileText size={20} />
          </div>
          <div className="flex-1">
            <h2 className="text-base font-semibold text-gray-900 dark:text-white">AI-Generated Investigation Summary</h2>
            <p className="text-gray-600 dark:text-gray-300 text-sm mt-2 leading-relaxed">{explain.investigation_summary}</p>
            <div className="mt-4 pt-4 border-t border-gray-50 dark:border-gray-700/50">
              <p className="text-xs text-gray-400 dark:text-gray-500 uppercase font-medium mb-1">Recommended Action</p>
              <p className="text-sm font-semibold text-gray-900 dark:text-white">{ACTION_BY_TIER[explain.risk_tier]}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function ScoreBar({ label, sublabel, value, color, bold }) {
  return (
    <div>
      <div className="flex justify-between items-baseline mb-1">
        <span className={`text-sm ${bold ? 'font-bold text-gray-900 dark:text-white' : 'font-medium text-gray-700 dark:text-gray-200'}`}>
          {label} {sublabel && <span className="text-xs text-gray-400 dark:text-gray-500 font-normal">({sublabel})</span>}
        </span>
        <span className={`text-sm ${bold ? 'font-bold' : 'font-semibold'}`} style={{ color }}>
          {value.toFixed(1)}
        </span>
      </div>
      <div className="w-full h-2.5 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${Math.min(100, value)}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}
