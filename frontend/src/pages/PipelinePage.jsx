import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { CheckCircle2, XCircle, Clock, ChevronRight, Cpu, Zap } from 'lucide-react'
import clsx from 'clsx'
import { pollJobStatus } from '../services/api'

const PIPELINE_STEPS = [
  'Dataset loaded',
  'Validation passed',
  'Problem type detected',
  'Exploratory Data Analysis',
  'Preprocessing & Feature Engineering',
  'AutoML Model Training',
  'Feature Importance Analysis',
  'Visualization Generation',
  'Orange3 Workflow Generation',
  'Pipeline complete',
]

export default function PipelinePage() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const [job, setJob] = useState(null)
  const [elapsed, setElapsed] = useState(0)

  // Timer
  useEffect(() => {
    const t = setInterval(() => setElapsed(e => e + 1), 1000)
    return () => clearInterval(t)
  }, [])

  // Poll
  useEffect(() => {
    if (!jobId) return
    const stop = pollJobStatus(jobId, (status) => {
      setJob(status)
      if (status.status === 'completed') {
        setTimeout(() => navigate(`/results/${jobId}`), 1200)
      }
    }, 1500)
    return stop
  }, [jobId, navigate])

  const fmt = (s) => `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`

  if (!job) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-10 h-10 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  const isFailed    = job.status === 'failed'
  const isCompleted = job.status === 'completed'
  const isRunning   = job.status === 'running' || job.status === 'pending'

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      {/* Header card */}
      <div className="card text-center space-y-4">
        <div className="flex items-center justify-center gap-3">
          {isCompleted ? (
            <CheckCircle2 className="w-10 h-10 text-emerald-400" />
          ) : isFailed ? (
            <XCircle className="w-10 h-10 text-rose-400" />
          ) : (
            <div className="relative w-10 h-10">
              <div className="absolute inset-0 rounded-full border-4 border-brand-500/20" />
              <div className="absolute inset-0 rounded-full border-4 border-brand-500 border-t-transparent animate-spin" />
              <Cpu className="absolute inset-0 m-auto w-4 h-4 text-brand-400" />
            </div>
          )}
          <div className="text-left">
            <h2 className="text-xl font-bold text-white">
              {isCompleted ? 'Pipeline Complete!' : isFailed ? 'Pipeline Failed' : 'Running AutoML Pipeline'}
            </h2>
            <p className="text-sm text-slate-400 font-mono">{jobId.slice(0, 18)}…</p>
          </div>
        </div>

        {/* Progress bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-xs text-slate-400">
            <span>{job.current_step}</span>
            <span className="font-mono">{job.progress}%</span>
          </div>
          <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full progress-bar-fill',
                isFailed ? 'bg-rose-500' : isCompleted ? 'bg-emerald-500' : 'bg-brand-500'
              )}
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>

        {/* Stats row */}
        <div className="flex justify-center gap-6 text-sm">
          <div className="flex items-center gap-2 text-slate-400">
            <Clock className="w-4 h-4" />
            <span className="font-mono">{fmt(elapsed)}</span>
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <Zap className="w-4 h-4" />
            <span>{job.steps_completed.length} / {PIPELINE_STEPS.length} steps</span>
          </div>
          <div>
            <span className={clsx('badge',
              isCompleted ? 'badge-green' : isFailed ? 'badge-red' : 'badge-blue'
            )}>
              {job.status}
            </span>
          </div>
        </div>

        {isFailed && job.error && (
          <div className="text-left bg-rose-500/10 border border-rose-500/30 rounded-xl p-4">
            <p className="text-xs font-semibold text-rose-400 mb-1">Error</p>
            <p className="text-sm text-rose-300 font-mono">{job.error}</p>
          </div>
        )}
      </div>

      {/* Steps list */}
      <div className="card">
        <h3 className="section-title">Pipeline Steps</h3>
        <div className="space-y-1">
          {PIPELINE_STEPS.map((step, idx) => {
            const done = job.steps_completed.includes(step)
            const active = !done && job.current_step.toLowerCase().includes(step.toLowerCase().split(' ')[0].toLowerCase())
            return (
              <div
                key={step}
                className={clsx(
                  'flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200',
                  done    ? 'bg-emerald-500/10' :
                  active  ? 'bg-brand-500/15 animate-pulse-slow' :
                  'opacity-40'
                )}
              >
                <div className={clsx(
                  'w-5 h-5 rounded-full flex items-center justify-center shrink-0',
                  done   ? 'bg-emerald-500' :
                  active ? 'bg-brand-500' :
                  'bg-surface-600'
                )}>
                  {done ? (
                    <CheckCircle2 className="w-3 h-3 text-white" />
                  ) : active ? (
                    <div className="w-2 h-2 bg-white rounded-full animate-ping" />
                  ) : (
                    <span className="text-[10px] text-slate-400 font-mono">{idx + 1}</span>
                  )}
                </div>
                <span className={clsx(
                  'text-sm',
                  done ? 'text-emerald-300' : active ? 'text-brand-300 font-medium' : 'text-slate-500'
                )}>
                  {step}
                </span>
                {active && (
                  <ChevronRight className="w-3 h-3 text-brand-400 ml-auto animate-bounce" />
                )}
              </div>
            )
          })}
        </div>
      </div>

      {isCompleted && (
        <div className="text-center">
          <p className="text-slate-400 text-sm">Redirecting to results…</p>
        </div>
      )}

      {isFailed && (
        <div className="text-center">
          <button onClick={() => navigate('/')} className="btn-secondary">
            ← Start Over
          </button>
        </div>
      )}
    </div>
  )
}
