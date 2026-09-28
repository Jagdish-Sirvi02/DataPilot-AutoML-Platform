import { Outlet, NavLink, useLocation } from 'react-router-dom'
import { Brain, Upload, BarChart3, FlaskConical, Github } from 'lucide-react'
import clsx from 'clsx'

const NAV = [
  { to: '/', label: 'Upload',   icon: Upload,     exact: true },
]

export default function Layout() {
  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <aside className="w-60 bg-surface-800 border-r border-surface-700 flex flex-col shrink-0">
        {/* Logo */}
        <div className="px-6 py-5 border-b border-surface-700">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-brand-500 rounded-lg flex items-center justify-center shrink-0">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-white text-sm leading-tight">DataPilot</h1>
              <p className="text-xs text-slate-500">AutoML Platform</p>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          <NavItem to="/" icon={Upload} label="Upload Dataset" end />
          <div className="pt-4 pb-1 px-3">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Resources</p>
          </div>
          <a
            href="https://orangedatamining.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-slate-400
                       hover:bg-surface-700 hover:text-slate-200 transition-all text-sm"
          >
            <FlaskConical className="w-4 h-4" />
            Orange3 Docs
          </a>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-slate-400
                       hover:bg-surface-700 hover:text-slate-200 transition-all text-sm"
          >
            <Github className="w-4 h-4" />
            GitHub
          </a>
        </nav>

        {/* Footer */}
        <div className="px-4 py-4 border-t border-surface-700">
          <div className="card-sm !p-3 text-center">
            <p className="text-xs text-slate-400 leading-relaxed">
              Upload a CSV → Get a trained ML pipeline + Orange3 workflow
            </p>
          </div>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Top bar */}
        <header className="h-14 bg-surface-800 border-b border-surface-700 flex items-center px-6 shrink-0">
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <BarChart3 className="w-4 h-4 text-brand-400" />
            <span className="text-slate-500">/</span>
            <Breadcrumb />
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="badge badge-green">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              API Online
            </span>
          </div>
        </header>

        {/* Page content */}
        <div className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}

function NavItem({ to, icon: Icon, label, end }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        clsx(
          'flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm transition-all duration-150',
          isActive
            ? 'bg-brand-500/20 text-brand-400 font-medium'
            : 'text-slate-400 hover:bg-surface-700 hover:text-slate-200'
        )
      }
    >
      <Icon className="w-4 h-4 shrink-0" />
      {label}
    </NavLink>
  )
}

function Breadcrumb() {
  const { pathname } = useLocation()
  const parts = pathname.split('/').filter(Boolean)
  if (parts.length === 0) return <span className="text-slate-300">Upload</span>
  if (parts[0] === 'pipeline') return (
    <>
      <span>Pipeline</span>
      <span className="text-slate-500">/</span>
      <span className="text-slate-300 font-mono text-xs">{parts[1]?.slice(0, 8)}…</span>
    </>
  )
  if (parts[0] === 'results') return (
    <>
      <span>Results</span>
      <span className="text-slate-500">/</span>
      <span className="text-slate-300 font-mono text-xs">{parts[1]?.slice(0, 8)}…</span>
    </>
  )
  return <span className="text-slate-300">{parts[0]}</span>
}
