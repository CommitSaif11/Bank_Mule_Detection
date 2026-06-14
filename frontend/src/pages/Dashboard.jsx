import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList, Cell } from 'recharts'
import { Users, AlertOctagon, AlertTriangle, ShieldAlert } from 'lucide-react'
import { getStats, getAlerts } from '../api/client'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import RiskBadge from '../components/RiskBadge'

const TIER_COLORS = { Low: '#22c55e', Medium: '#eab308', High: '#f97316', Critical: '#ef4444' }

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getStats(), getAlerts()])
      .then(([statsRes, alertsRes]) => {
        setStats(statsRes.data)
        setAlerts(alertsRes.data)
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <Loading />
  if (error) return <ErrorMessage message={error} />

  const chartData = [
    { tier: 'Low', count: stats.low, fill: TIER_COLORS.Low },
    { tier: 'Medium', count: stats.medium, fill: TIER_COLORS.Medium },
    { tier: 'High', count: stats.high, fill: TIER_COLORS.High },
    { tier: 'Critical', count: stats.critical, fill: TIER_COLORS.Critical },
  ]

  const topCritical = alerts.filter((a) => a.risk_tier === 'Critical').slice(0, 10)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Fraud Intelligence Dashboard</h1>
        <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">Batch-mode scoring across the full account population</p>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-4 gap-5">
        <MetricCard
          icon={<Users size={20} />}
          label="Total Accounts"
          value={stats.total_accounts.toLocaleString()}
          tone="gray"
        />
        <MetricCard
          icon={<AlertOctagon size={20} />}
          label="Critical Alerts"
          value={stats.critical.toLocaleString()}
          tone="red"
        />
        <MetricCard
          icon={<AlertTriangle size={20} />}
          label="High Alerts"
          value={stats.high.toLocaleString()}
          tone="orange"
        />
        <MetricCard
          icon={<ShieldAlert size={20} />}
          label="Fraud Detected"
          value={stats.fraud_detected.toLocaleString()}
          sub={`${stats.fraud_detected} accounts flagged`}
          tone="red"
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-5 gap-5">
        <div className="col-span-3 bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4">Risk Tier Distribution</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
              <XAxis dataKey="tier" tick={{ fontSize: 13 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fontSize: 13 }} axisLine={false} tickLine={false} />
              <Tooltip />
              <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                <LabelList dataKey="count" position="top" style={{ fontWeight: 600, fontSize: 13 }} />
                {chartData.map((entry, i) => (
                  <Cell key={i} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="col-span-2 bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
          <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4">Detection Performance</h2>
          <dl className="space-y-3 text-sm">
            <StatRow label="Accounts analysed" value="9,082" />
            <StatRow label="Fraud accounts identified" value="81" />
            <StatRow label="Detection rate" value="96.3% (78/81 in High+Critical)" highlight />
            <StatRow label="Mean risk score (fraud)" value="81.7" />
            <StatRow label="Mean risk score (legit)" value="8.3" />
            <StatRow label="False negatives" value="1 account" />
          </dl>
        </div>
      </div>

      {/* Top critical accounts */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4">Top Critical Accounts</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-gray-400 dark:text-gray-500 border-b border-gray-100 dark:border-gray-700">
              <th className="py-2 font-medium">Rank</th>
              <th className="py-2 font-medium">Account ID</th>
              <th className="py-2 font-medium">Risk Score</th>
              <th className="py-2 font-medium">Typology</th>
              <th className="py-2 font-medium text-right">Action</th>
            </tr>
          </thead>
          <tbody>
            {topCritical.map((acc, i) => (
              <tr key={acc.account_index} className="border-b border-gray-50 dark:border-gray-700/50 border-l-4 border-l-red-500">
                <td className="py-3 pl-3 text-gray-500 dark:text-gray-400">{i + 1}</td>
                <td className="py-3 font-medium text-gray-900 dark:text-white">{acc.account_index}</td>
                <td className="py-3 font-semibold">{acc.risk_score.toFixed(1)}</td>
                <td className="py-3 text-gray-600 dark:text-gray-300">{acc.typology_label}</td>
                <td className="py-3 text-right">
                  <Link
                    to={`/account/${acc.account_index}`}
                    className="inline-block px-3 py-1.5 rounded-lg bg-blue-500 text-white text-xs font-semibold hover:bg-blue-600 transition-colors"
                  >
                    Investigate
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Model performance */}
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-6">
        <h2 className="text-base font-semibold text-gray-900 dark:text-white mb-4">Model Performance</h2>
        <div className="grid grid-cols-4 gap-5 text-sm">
          <ModelStat label="Model Used" value="LightGBM" />
          <ModelStat label="Training Samples" value="7,265" />
          <ModelStat label="Features Used" value="176" />
          <ModelStat label="Class Imbalance Handled" value="SMOTE (111:1 ratio)" />
          <ModelStat label="CV Recall" value="72.8% ± 8.6%" />
          <ModelStat label="CV Precision" value="98.7% ± 2.7%" />
          <ModelStat label="AUC-ROC" value="99.1% ± 0.5%" />
          <ModelStat label="AUC-PR" value="92.9% ± 4.2%" />
        </div>
        <p className="text-xs text-gray-400 dark:text-gray-500 mt-4 pt-4 border-t border-gray-50 dark:border-gray-700/50">
          Evaluated on real unseen data with no synthetic augmentation in test set.
        </p>
      </div>
    </div>
  )
}

function ModelStat({ label, value }) {
  return (
    <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg px-4 py-3">
      <p className="text-gray-400 dark:text-gray-500 text-xs uppercase mb-1">{label}</p>
      <p className="font-bold text-gray-900 dark:text-white">{value}</p>
    </div>
  )
}

function MetricCard({ icon, label, value, sub, tone }) {
  const tones = {
    gray: 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300',
    red: 'bg-red-50 text-critical',
    orange: 'bg-orange-50 text-high',
  }
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm p-5">
      <div className={`inline-flex items-center justify-center w-9 h-9 rounded-lg mb-3 ${tones[tone]}`}>
        {icon}
      </div>
      <p className="text-gray-400 dark:text-gray-500 text-xs font-medium uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-bold text-gray-900 dark:text-white mt-1">{value}</p>
      {sub && <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">{sub}</p>}
    </div>
  )
}

function StatRow({ label, value, highlight }) {
  return (
    <div className="flex items-center justify-between py-1.5 border-b border-gray-50 dark:border-gray-700/50 last:border-0">
      <dt className="text-gray-500 dark:text-gray-400">{label}</dt>
      <dd className={`font-semibold ${highlight ? 'text-blue-600' : 'text-gray-900 dark:text-white'}`}>{value}</dd>
    </div>
  )
}
