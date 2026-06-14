import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertTriangle } from 'lucide-react'
import { getAlerts } from '../api/client'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import RiskBadge from '../components/RiskBadge'

const TIER_COLORS = { Low: '#22c55e', Medium: '#eab308', High: '#f97316', Critical: '#ef4444' }

export default function Alerts() {
  const [alerts, setAlerts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tierFilter, setTierFilter] = useState('All')
  const [typologyFilter, setTypologyFilter] = useState('All')
  const [search, setSearch] = useState('')
  const navigate = useNavigate()

  useEffect(() => {
    getAlerts()
      .then((res) => setAlerts(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} />

  const filtered = alerts.filter((a) => {
    if (tierFilter !== 'All' && a.risk_tier !== tierFilter) return false
    if (typologyFilter !== 'All' && a.typology_label !== typologyFilter) return false
    if (search && !String(a.account_index).includes(search)) return false
    return true
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <AlertTriangle className="text-high" size={26} />
          Alert Console
        </h1>
        <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">78 accounts require immediate investigation</p>
      </div>

      {/* Filter bar */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-4 flex flex-wrap items-center gap-3">
        <select
          value={tierFilter}
          onChange={(e) => setTierFilter(e.target.value)}
          className="border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        >
          <option>All</option>
          <option>Critical</option>
          <option>High</option>
        </select>

        <select
          value={typologyFilter}
          onChange={(e) => setTypologyFilter(e.target.value)}
          className="border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
        >
          <option value="All">All Typologies</option>
          <option>Complicit Mule</option>
          <option>Recruited Mule</option>
          <option>Exploited Mule</option>
          <option>Low Risk</option>
        </select>

        <input
          type="text"
          placeholder="Search by account ID..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-lg px-3 py-2 text-sm flex-1 min-w-[200px] focus:outline-none focus:ring-2 focus:ring-blue-400"
        />

        <span className="text-sm text-gray-400 dark:text-gray-500 ml-auto">
          Showing {filtered.length} of {alerts.length} alerts
        </span>
      </div>

      {/* Table */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-gray-400 dark:text-gray-500 border-b border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-700/30">
              <th className="py-3 px-4 font-medium">#</th>
              <th className="py-3 px-4 font-medium">Account ID</th>
              <th className="py-3 px-4 font-medium">Risk Score</th>
              <th className="py-3 px-4 font-medium">Risk Tier</th>
              <th className="py-3 px-4 font-medium">Typology</th>
              <th className="py-3 px-4 font-medium">ML Probability</th>
              <th className="py-3 px-4 font-medium text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((a, i) => (
              <tr
                key={a.account_index}
                onClick={() => navigate(`/account/${a.account_index}`)}
                className={`cursor-pointer border-b border-gray-50 dark:border-gray-700/50 hover:bg-blue-50/50 dark:hover:bg-blue-900/20 transition-colors ${
                  i % 2 === 0 ? 'bg-white dark:bg-gray-800' : 'bg-gray-50/30 dark:bg-gray-800/60'
                }`}
              >
                <td className="py-3 px-4 text-gray-500 dark:text-gray-400">{i + 1}</td>
                <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">{a.account_index}</td>
                <td className="py-3 px-4">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold w-10">{a.risk_score.toFixed(1)}</span>
                    <div className="w-20 h-1.5 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${a.risk_score}%`,
                          backgroundColor: TIER_COLORS[a.risk_tier],
                        }}
                      />
                    </div>
                  </div>
                </td>
                <td className="py-3 px-4"><RiskBadge tier={a.risk_tier} /></td>
                <td className="py-3 px-4 text-gray-600 dark:text-gray-300">{a.typology_label}</td>
                <td className="py-3 px-4 font-medium text-gray-700 dark:text-gray-200">{a.ml_score.toFixed(1)}%</td>
                <td className="py-3 px-4 text-right">
                  <button
                    onClick={(e) => {
                      e.stopPropagation()
                      navigate(`/account/${a.account_index}`)
                    }}
                    className="px-3 py-1.5 rounded-lg bg-blue-500 text-white text-xs font-semibold hover:bg-blue-600 transition-colors"
                  >
                    Investigate
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
