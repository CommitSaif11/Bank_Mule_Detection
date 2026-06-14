import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Alerts from './pages/Alerts'
import Accounts from './pages/Accounts'
import Account from './pages/Account'
import Upload from './pages/Upload'
import LiveScore from './pages/LiveScore'

export default function App() {
  return (
    <div className="flex min-h-screen bg-[#f8fafc] dark:bg-[#0b0f19]">
      <Navbar />
      <main className="flex-1 ml-[240px] p-8">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/accounts" element={<Accounts />} />
          <Route path="/account/:id" element={<Account />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/live" element={<LiveScore />} />
        </Routes>
      </main>
    </div>
  )
}
