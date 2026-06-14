import { useEffect, useState } from 'react'
import { Zap, Info, RotateCcw, TrendingUp, TrendingDown } from 'lucide-react'
import { getLiveSample, postLiveScore } from '../api/client'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import RiskGauge from '../components/RiskGauge'
import RiskBadge from '../components/RiskBadge'

const FEATURES = [
  { key: 'F115', hint: 'Transaction concentration ratio' },
  { key: 'F321', hint: 'Normalized frequency metric' },
  { key: 'F527', hint: 'Velocity ratio' },
  { key: 'F531', hint: 'Amount metric' },
  { key: 'F670', hint: 'Binary flag (0 or 1)' },
  { key: 'F1692', hint: 'Linked entities count' },
  { key: 'F2082', hint: 'Behavioral rate' },
  { key: 'F2122', hint: 'Activity ratio' },
  { key: 'F2582', hint: 'Flow metric' },
  { key: 'F2678', hint: 'Transaction amount' },
  { key: 'F2737', hint: 'Ratio metric' },
  { key: 'F2956', hint: 'Activity days' },
  { key: 'F3043', hint: 'Volume metric' },
  { key: 'F3836', hint: 'Balance indicator' },
  { key: 'F3887', hint: 'Account tenure (months)' },
  { key: 'F3889', hint: 'Observation window', options: ['G365D', 'L365D', 'L180D', 'L90D', 'L31D', 'L14D', 'L7D'] },
  { key: 'F3891', hint: 'Occupation', options: ['selfemployed', 'salaried', 'student', 'agriculture', 'housewife', 'others', 'retired'] },
  { key: 'F3894', hint: 'Account holder age' },
]

export default function LiveScore() {
  const [sample, setSample] = useState(null)
  const [highRiskSample, setHighRiskSample] = useState(null)
  const [form, setForm] = useState(null)
  const [result, setResult] = useState(null)
  const [scoring, setScoring] = useState(false)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)
  const [history, setHistory] = useState([])

  useEffect(() => {
    getLiveSample()
      .then((res) => {
        setSample(res.data.sample_normal)
        setHighRiskSample(res.data.sample_high_risk)
        setForm(res.data.sample_normal)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} />

  const setField = (key, value) => setForm((prev) => ({ ...prev, [key]: value }))

  const applyPreset = (preset) => setForm({ ...preset })
  const resetToSample = () => setForm(sample)

  const handleScore = async (overrideForm) => {
    const payload = {}
    for (const { key } of FEATURES) {
      const source = overrideForm || form
      payload[key] = source?.[key] ?? sample?.[key]
    }
    setScoring(true)
    setError(null)
    try {
      const res = await postLiveScore(payload)
      setResult(res.data)
      setHistory((prev) => [{ features: payload, result: res.data, time: new Date() }, ...prev].slice(0, 10))
    } catch (err) {
      setError(err.response?.data?.detail || err.message)
    } finally {
      setScoring(false)
    }
  }

  const restoreFromHistory = (entry) => {
    setForm(entry.features)
    setResult(entry.result)
  }

  const maxAbsShap = result
    ? Math.max(...result.top_risk_factors.map((f) => Math.abs(f.shap_value)), 0.001)
    : 1

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2.5">
          <Zap className="text-blue-500" size={26} />
          Live Account Scorer
        </h1>
        <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
          Score any account instantly using the trained MuleNet model. Simulates real-time API feed integration.
        </p>
      </div>

      <div className="flex items-start gap-3 bg-blue-50 dark:bg-blue-500/10 text-blue-700 dark:text-blue-300 rounded-xl px-4 py-3 text-sm">
        <Info size={18} className="mt-0.5 flex-shrink-0" />
        <p>
          Live feed mode — scores are generated in milliseconds using the trained LightGBM model. In production, this
          endpoint connects to your transaction monitoring system.
        </p>
      </div>

      <div className="flex items-start gap-3 bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 rounded-xl px-4 py-3 text-xs font-mono">
        <code className="flex-1">
          API Endpoint: POST /api/score/live — Average response: &lt;50ms
          <br />
          Production integration: Connect to your transaction monitoring system or fraud alert feed via this endpoint.
        </code>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-500/10 text-red-600 dark:text-red-400 rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* Input form */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white">Account Features</h2>
          <div className="flex gap-2">
            <button
              onClick={() => applyPreset(highRiskSample)}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-red-50 dark:bg-red-500/10 text-critical hover:bg-red-100 dark:hover:bg-red-500/20 transition-colors"
            >
              Load High Risk Profile
            </button>
            <button
              onClick={() => applyPreset(sample)}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-green-50 dark:bg-green-500/10 text-low hover:bg-green-100 dark:hover:bg-green-500/20 transition-colors"
            >
              Load Low Risk Profile
            </button>
            <button
              onClick={resetToSample}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              <RotateCcw size={12} />
              Reset to Sample
            </button>
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          {FEATURES.map(({ key, hint, options }) => (
            <div key={key}>
              <label className="block text-sm font-semibold text-gray-900 dark:text-white mb-1">{key}</label>
              {options ? (
                <select
                  value={form[key] ?? ''}
                  onChange={(e) => setField(key, e.target.value)}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-700 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {options.map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  value={form[key] ?? ''}
                  onChange={(e) => setField(key, e.target.value === '' ? '' : Number(e.target.value))}
                  className="w-full px-3 py-2 rounded-lg border border-gray-200 dark:border-gray-600 bg-white dark:bg-gray-700 text-sm text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              )}
              <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">{hint}</p>
            </div>
          ))}
        </div>

        <button
          onClick={() => handleScore()}
          disabled={scoring}
          className="mt-6 w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-blue-500 text-white text-sm font-bold hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {scoring ? (
            <>
              <div className="h-4 w-4 border-2 border-white/40 border-t-white rounded-full animate-spin" />
              Scoring…
            </>
          ) : (
            <>
              <Zap size={16} />
              Score This Account
            </>
          )}
        </button>

        {result && (result.risk_tier === 'Critical' || result.risk_tier === 'High') && (
          <div className="mt-4 flex items-center gap-2 bg-red-50 dark:bg-red-500/10 text-critical rounded-lg px-4 py-3 text-sm font-bold">
            HIGH RISK ACCOUNT DETECTED — Recommend immediate investigation
          </div>
        )}
        {result && result.risk_tier === 'Low' && (
          <div className="mt-4 flex items-center gap-2 bg-green-50 dark:bg-green-500/10 text-low rounded-lg px-4 py-3 text-sm font-bold">
            LOW RISK — Normal account behavior detected
          </div>
        )}
      </div>

      {/* Result panel */}
      {result && (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6 grid grid-cols-3 gap-6 items-center">
            <div className="flex justify-center">
              <RiskGauge score={result.risk_score} tier={result.risk_tier} size={180} />
            </div>
            <div className="col-span-2 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Risk Tier</p>
                  <RiskBadge tier={result.risk_tier} />
                </div>
                <div>
                  <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Typology</p>
                  <p className="font-semibold text-gray-900 dark:text-white">{result.typology}</p>
                </div>
                <div>
                  <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">ML Fraud Probability</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">
                    {(result.ml_fraud_probability * 100).toFixed(2)}%
                  </p>
                </div>
                <div>
                  <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Anomaly Score</p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-white">{result.anomaly_score.toFixed(1)}</p>
                </div>
              </div>
              <div className="pt-3 border-t border-gray-50 dark:border-gray-700/50">
                <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">Processing Time</p>
                <p className="text-2xl font-bold text-green-500">{result.processing_time_ms.toFixed(2)} ms</p>
              </div>
            </div>
          </div>

          {/* SHAP factors */}
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
            <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-1">Top Risk Factors</h2>
            <p className="text-gray-500 dark:text-gray-400 text-sm mb-4">SHAP feature impact for this account</p>
            <div className="space-y-4">
              {result.top_risk_factors.map((f) => {
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
            <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-3">Investigation Summary</h2>
            <blockquote className="border-l-4 border-blue-400 bg-gray-50 dark:bg-gray-700/50 rounded-r-lg px-4 py-3 text-sm text-gray-700 dark:text-gray-300 italic">
              "{result.investigation_summary}"
            </blockquote>
          </div>
        </>
      )}

      {/* Score history */}
      {history.length > 0 && (
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-1">Score History</h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mb-4">Last {history.length} scores from this session — click a row to restore</p>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-400 dark:text-gray-500 text-xs uppercase border-b border-gray-100 dark:border-gray-700">
                <th className="py-2 pr-3">#</th>
                <th className="py-2 pr-3">Key Features</th>
                <th className="py-2 pr-3">Risk Score</th>
                <th className="py-2 pr-3">Tier</th>
                <th className="py-2 pr-3">Time</th>
              </tr>
            </thead>
            <tbody>
              {history.map((entry, i) => (
                <tr
                  key={i}
                  onClick={() => restoreFromHistory(entry)}
                  className="border-b border-gray-50 dark:border-gray-700/50 last:border-0 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/40 transition-colors"
                >
                  <td className="py-2 pr-3 text-gray-400 dark:text-gray-500">{history.length - i}</td>
                  <td className="py-2 pr-3 text-gray-700 dark:text-gray-300">
                    F670={entry.features.F670}, F3891={entry.features.F3891}, F3894={entry.features.F3894}
                  </td>
                  <td className="py-2 pr-3 font-semibold text-gray-900 dark:text-white flex items-center gap-1">
                    {entry.result.risk_score.toFixed(1)}
                    {entry.result.risk_score >= 50 ? (
                      <TrendingUp size={14} className="text-critical" />
                    ) : (
                      <TrendingDown size={14} className="text-low" />
                    )}
                  </td>
                  <td className="py-2 pr-3">
                    <RiskBadge tier={entry.result.risk_tier} />
                  </td>
                  <td className="py-2 pr-3 text-gray-400 dark:text-gray-500">{entry.time.toLocaleTimeString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
