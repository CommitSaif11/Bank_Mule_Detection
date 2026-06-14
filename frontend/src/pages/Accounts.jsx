import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Users } from 'lucide-react'
import { getAccounts } from '../api/client'
import Loading from '../components/Loading'
import ErrorMessage from '../components/ErrorMessage'
import RiskBadge from '../components/RiskBadge'

const PAGE_SIZE = 50

export default function Accounts() {
  const [accounts, setAccounts] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [tierFilter, setTierFilter] = useState('All')
  const [typologyFilter, setTypologyFilter] = useState('All')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(1)
  const navigate = useNavigate()

  useEffect(() => {
    setLoading(true)
    const params = {}
    if (tierFilter !== 'All') params.tier = tierFilter
    if (typologyFilter !== 'All') params.typology = typologyFilter
    getAccounts(params)
      .then((res) => setAccounts(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [tierFilter, typologyFilter])

  useEffect(() => setPage(1), [tierFilter, typologyFilter, search])

  if (error) return <ErrorMessage message={error} />

  const filtered = accounts.filter((a) =>
    search ? String(a.account_index).includes(search) : true
  )
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const pageItems = filtered.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Users className="text-blue-500" size={26} />
          Account Registry
        </h1>
        <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">{accounts.length.toLocaleString()} accounts in current view</p>
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
          <option>Medium</option>
          <option>Low</option>
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
      </div>

      {loading ? (
        <Loading />
      ) : (
        <>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-gray-400 dark:text-gray-500 border-b border-gray-100 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-700/30">
                  <th className="py-3 px-4 font-medium">Account ID</th>
                  <th className="py-3 px-4 font-medium">Risk Score</th>
                  <th className="py-3 px-4 font-medium">Risk Tier</th>
                  <th className="py-3 px-4 font-medium">Typology</th>
                  <th className="py-3 px-4 font-medium">True Label</th>
                  <th className="py-3 px-4 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {pageItems.map((a, i) => (
                  <tr
                    key={a.account_index}
                    onClick={() => navigate(`/account/${a.account_index}`)}
                    className={`cursor-pointer border-b border-gray-50 dark:border-gray-700/50 hover:bg-blue-50/50 dark:hover:bg-blue-900/20 transition-colors ${
                      i % 2 === 0 ? 'bg-white dark:bg-gray-800' : 'bg-gray-50/30 dark:bg-gray-800/60'
                    }`}
                  >
                    <td className="py-3 px-4 font-medium text-gray-900 dark:text-white">{a.account_index}</td>
                    <td className="py-3 px-4 font-semibold">{a.risk_score.toFixed(1)}</td>
                    <td className="py-3 px-4"><RiskBadge tier={a.risk_tier} /></td>
                    <td className="py-3 px-4 text-gray-600 dark:text-gray-300">{a.typology_label}</td>
                    <td className="py-3 px-4">
                      {a.true_label === 1 ? (
                        <span className="text-critical font-semibold">Fraud</span>
                      ) : (
                        <span className="text-low font-semibold">Legit</span>
                      )}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          navigate(`/account/${a.account_index}`)
                        }}
                        className="px-3 py-1.5 rounded-lg bg-blue-500 text-white text-xs font-semibold hover:bg-blue-600 transition-colors"
                      >
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
            <span>
              Page {page} of {totalPages} ({filtered.length.toLocaleString()} accounts)
            </span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 disabled:opacity-40 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Previous
              </button>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="px-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 disabled:opacity-40 hover:bg-gray-50 dark:hover:bg-gray-700"
              >
                Next
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
