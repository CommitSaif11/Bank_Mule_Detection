import { NavLink } from 'react-router-dom'
import { Shield, LayoutDashboard, AlertTriangle, Users, Sun, Moon, Upload, Zap } from 'lucide-react'
import { useTheme } from '../context/ThemeContext'

const links = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/alerts', label: 'Alerts', icon: AlertTriangle },
  { to: '/accounts', label: 'Accounts', icon: Users },
  { to: '/upload', label: 'Upload Dataset', icon: Upload },
  { to: '/live', label: 'Live Scorer', icon: Zap },
]

export default function Navbar() {
  const { theme, toggleTheme } = useTheme()

  return (
    <aside className="w-[240px] min-h-screen bg-[#0f172a] text-white flex flex-col fixed left-0 top-0">
      <div className="px-6 py-6 border-b border-white/10">
        <div className="flex items-center gap-2">
          <Shield className="text-blue-400" size={28} />
          <span className="text-xl font-bold tracking-tight">MuleNet</span>
        </div>
        <p className="text-xs text-gray-400 mt-1">Fraud Intelligence Platform</p>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {links.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-blue-500/15 text-blue-400'
                  : 'text-gray-300 hover:bg-white/5 hover:text-white'
              }`
            }
          >
            <Icon size={18} />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="px-6 py-4 border-t border-white/10 text-xs text-gray-400 space-y-3">
        <button
          onClick={toggleTheme}
          className="flex items-center gap-2 w-full px-2 py-2 rounded-lg text-gray-300 hover:bg-white/5 hover:text-white transition-colors"
        >
          {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          {theme === 'dark' ? 'Light Mode' : 'Dark Mode'}
        </button>
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-green-400"></span>
          API Connected
        </div>
        <p>v1.0.0 — Batch Mode</p>
      </div>
    </aside>
  )
}
