import { useEffect, useState } from 'react'

function getColor(score) {
  if (score <= 30) return '#22c55e'
  if (score <= 60) return '#eab308'
  if (score <= 80) return '#f97316'
  return '#ef4444'
}

export default function RiskGauge({ score, tier, size = 200 }) {
  const [animated, setAnimated] = useState(0)

  useEffect(() => {
    const target = Math.max(0, Math.min(100, score))
    let frame
    const start = performance.now()
    const duration = 900

    const step = (now) => {
      const progress = Math.min(1, (now - start) / duration)
      setAnimated(target * progress)
      if (progress < 1) frame = requestAnimationFrame(step)
    }
    frame = requestAnimationFrame(step)
    return () => cancelAnimationFrame(frame)
  }, [score])

  const radius = (size - 20) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (animated / 100) * circumference
  const color = getColor(score)

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="-rotate-90">
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke="currentColor"
            className="text-gray-200 dark:text-gray-700"
            strokeWidth="14"
            fill="none"
          />
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            stroke={color}
            strokeWidth="14"
            fill="none"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{ transition: 'stroke 0.3s' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-5xl font-bold text-gray-900 dark:text-white">{Math.round(animated)}</span>
          <span className="text-sm font-medium text-gray-400 mt-1">Risk Score</span>
        </div>
      </div>
      <div
        className="mt-3 px-4 py-1 rounded-full text-sm font-semibold"
        style={{ backgroundColor: `${color}1A`, color }}
      >
        {tier}
      </div>
    </div>
  )
}
