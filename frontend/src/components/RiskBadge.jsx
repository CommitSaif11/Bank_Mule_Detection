const TIER_STYLES = {
  Critical: 'bg-critical text-white',
  High: 'bg-high text-white',
  Medium: 'bg-medium text-gray-900',
  Low: 'bg-low text-white',
}

export default function RiskBadge({ tier }) {
  const style = TIER_STYLES[tier] || 'bg-gray-300 text-gray-800'

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${style}`}>
      {tier === 'Critical' && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-white opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-white"></span>
        </span>
      )}
      {tier}
    </span>
  )
}
