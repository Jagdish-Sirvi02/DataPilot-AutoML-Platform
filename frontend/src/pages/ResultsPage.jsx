import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import {
  Trophy, Download, BarChart2, Activity, Target,
  ArrowLeft, Image, AlertCircle, ChevronDown, ChevronUp,
  Star, Layers, TrendingUp
} from 'lucide-react'
import clsx from 'clsx'
import { Bar } from 'react-chartjs-2'
import {
  Chart as ChartJS, CategoryScale, LinearScale,
  BarElement, Title, Tooltip, Legend
} from 'chart.js'
import { getModelResults, getWorkflowInfo, downloadWorkflowUrl } from '../services/api'

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)

export default function ResultsPage() {
  const { jobId } = useParams()
  const navigate  = useNavigate()
  const [results,  setResults]  = useState(null)
  const [workflow, setWorkflow] = useState(null)
  const [loading,  setLoading]  = useState(true)
  const [activeTab, setActiveTab] = useState('overview')
  const [expandedModel, setExpandedModel] = useState(null)

  useEffect(() => {
    if (!jobId) return
    Promise.all([
      getModelResults(jobId),
      getWorkflowInfo(jobId),
    ]).then(([res, wf]) => {
      setResults(res)
      setWorkflow(wf)
    }).catch(console.error)
      .finally(() => setLoading(false))
  }, [jobId])

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="w-10 h-10 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )

  if (!results) return (
    <div className="text-center py-20 text-slate-400">
      <AlertCircle className="w-10 h-10 mx-auto mb-3" />
      <p>Results not found. The job may still be running.</p>
      <button onClick={() => navigate('/')} className="btn-secondary mt-4">← Back</button>
    </div>
  )

  const isClassification = results.problem_type === 'classification'
  const tabs = ['overview', 'models', 'features', 'visualizations']

  return (
    <div className="max-w-5xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <button onClick={() => navigate('/')} className="btn-ghost flex items-center gap-1.5 text-sm mb-3">
            <ArrowLeft className="w-3.5 h-3.5" /> New Analysis
          </button>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Trophy className="w-6 h-6 text-amber-400" />
            AutoML Results
          </h2>
          <p className="text-slate-400 text-sm mt-1 font-mono">{jobId.slice(0, 24)}…</p>
        </div>
        {workflow?.workflow_available && (
          <a
            href={downloadWorkflowUrl(jobId)}
            download
            className="btn-primary flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Download .ows Workflow
          </a>
        )}
      </div>

      {/* Hero metrics */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <HeroCard
          label="Best Model"
          value={results.best_model}
          icon={<Star className="w-4 h-4 text-amber-400" />}
          accent="amber"
          small
        />
        <HeroCard
          label={isClassification ? 'Best F1 Score' : 'Best R² Score'}
          value={(results.best_score * 100).toFixed(2) + '%'}
          icon={<TrendingUp className="w-4 h-4 text-emerald-400" />}
          accent="emerald"
        />
        <HeroCard
          label="Models Trained"
          value={results.models.length}
          icon={<Layers className="w-4 h-4 text-brand-400" />}
          accent="brand"
        />
        <HeroCard
          label="Task Type"
          value={results.problem_type}
          icon={<Target className="w-4 h-4 text-cyan-400" />}
          accent="cyan"
          small
        />
      </div>

      {/* Orange workflow banner */}
      {workflow?.workflow_available && (
        <div className="card !p-4 border-brand-500/30 bg-brand-500/5 flex items-center gap-4">
          <div className="w-10 h-10 bg-brand-500/20 rounded-xl flex items-center justify-center shrink-0">
            <BarChart2 className="w-5 h-5 text-brand-400" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="font-semibold text-brand-300 text-sm">Orange3 Workflow Ready</p>
            <p className="text-xs text-slate-400 mt-0.5">{workflow.instructions}</p>
          </div>
          <a href={downloadWorkflowUrl(jobId)} download className="btn-primary !py-2 shrink-0">
            <Download className="w-3.5 h-3.5" />
          </a>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-surface-700">
        <div className="flex gap-1">
          {tabs.map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={clsx(
                'px-4 py-2.5 text-sm font-medium capitalize transition-all border-b-2 -mb-px',
                activeTab === tab
                  ? 'border-brand-500 text-brand-300'
                  : 'border-transparent text-slate-400 hover:text-slate-200'
              )}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      {activeTab === 'overview' && <OverviewTab results={results} isClassification={isClassification} />}
      {activeTab === 'models'   && <ModelsTab results={results} isClassification={isClassification} expandedModel={expandedModel} setExpandedModel={setExpandedModel} />}
      {activeTab === 'features' && <FeaturesTab results={results} />}
      {activeTab === 'visualizations' && <VisualizationsTab results={results} />}
    </div>
  )
}

// ── Sub-components ─────────────────────────────────────────────────────────

function HeroCard({ label, value, icon, accent, small }) {
  const colors = {
    amber:   'text-amber-300',
    emerald: 'text-emerald-300',
    brand:   'text-brand-300',
    cyan:    'text-cyan-300',
  }
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-2">{icon}<span className="metric-label">{label}</span></div>
      <p className={clsx(small ? 'text-lg' : 'text-2xl', 'font-bold', colors[accent], 'truncate')}>{value}</p>
    </div>
  )
}

function OverviewTab({ results, isClassification }) {
  const best = results.models[0]
  if (!best) return null

  const metrics = isClassification
    ? [
        { label: 'Accuracy',  value: best.accuracy,  color: '#6366f1' },
        { label: 'Precision', value: best.precision, color: '#22d3ee' },
        { label: 'Recall',    value: best.recall,    color: '#10b981' },
        { label: 'F1 Score',  value: best.f1_score,  color: '#f59e0b' },
        { label: 'ROC AUC',   value: best.roc_auc,   color: '#a78bfa' },
      ]
    : [
        { label: 'R² Score', value: best.r2,   color: '#6366f1' },
        { label: 'RMSE',     value: best.rmse, color: '#f43f5e', lower: true },
        { label: 'MAE',      value: best.mae,  color: '#f59e0b', lower: true },
      ]

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="card">
        <h3 className="section-title">Best Model: {best.model_name}</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {metrics.map(m => m.value != null && (
            <div key={m.label} className="card-sm text-center">
              <p className="text-xl font-bold" style={{ color: m.color }}>
                {m.lower ? m.value.toFixed(4) : (m.value * 100).toFixed(2) + '%'}
              </p>
              <p className="metric-label">{m.label}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-3">
          <div className="card-sm">
            <p className="text-base font-bold text-white">{(best.cv_score_mean * 100).toFixed(2)}% ± {(best.cv_score_std * 100).toFixed(2)}%</p>
            <p className="metric-label">Cross-validation Score</p>
          </div>
          <div className="card-sm">
            <p className="text-base font-bold text-white">{best.training_time_sec}s</p>
            <p className="metric-label">Training Time</p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3 className="section-title">Preprocessing Steps Applied</h3>
        <ol className="space-y-2">
          {results.preprocessing_steps.map((step, i) => (
            <li key={i} className="flex items-start gap-3 text-sm text-slate-300">
              <span className="w-5 h-5 rounded-full bg-brand-500/20 text-brand-400 text-xs flex items-center justify-center shrink-0 mt-0.5">
                {i + 1}
              </span>
              {step}
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

function ModelsTab({ results, isClassification, expandedModel, setExpandedModel }) {
  const chartData = {
    labels: results.models.map(m => m.model_name),
    datasets: [{
      label: isClassification ? 'F1 Score' : 'R² Score',
      data: results.models.map(m => isClassification ? (m.f1_score || 0) : (m.r2 || 0)),
      backgroundColor: results.models.map((_, i) =>
        i === 0 ? 'rgba(99, 102, 241, 0.85)' : 'rgba(99, 102, 241, 0.35)'
      ),
      borderColor: results.models.map((_, i) =>
        i === 0 ? '#6366f1' : 'rgba(99, 102, 241, 0.5)'
      ),
      borderWidth: 2,
      borderRadius: 8,
    }],
  }

  const chartOpts = {
    responsive: true,
    plugins: {
      legend: { display: false },
      tooltip: { callbacks: { label: ctx => ` ${(ctx.raw * 100).toFixed(2)}%` } },
    },
    scales: {
      x: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
      y: {
        grid: { color: '#334155' }, ticks: { color: '#94a3b8' },
        min: 0, max: 1,
        ticks: { callback: v => `${(v * 100).toFixed(0)}%`, color: '#94a3b8' }
      },
    },
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="card">
        <h3 className="section-title">Model Comparison</h3>
        <div style={{ height: 260 }}>
          <Bar data={chartData} options={{ ...chartOpts, maintainAspectRatio: false }} />
        </div>
      </div>

      <div className="space-y-3">
        {results.models.map((model, idx) => (
          <div key={model.model_name} className={clsx('card cursor-pointer transition-all', idx === 0 && 'border-brand-500/40')}>
            <div
              className="flex items-center justify-between"
              onClick={() => setExpandedModel(expandedModel === model.model_name ? null : model.model_name)}
            >
              <div className="flex items-center gap-3">
                {idx === 0 && <Star className="w-4 h-4 text-amber-400" />}
                <span className="font-semibold text-slate-200">{model.model_name}</span>
                {idx === 0 && <span className="badge badge-green text-[10px]">Best</span>}
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm text-slate-400">
                  {isClassification
                    ? `F1: ${(model.f1_score * 100).toFixed(2)}%`
                    : `R²: ${(model.r2 * 100).toFixed(2)}%`
                  }
                </span>
                <span className="text-xs text-slate-500">{model.training_time_sec}s</span>
                {expandedModel === model.model_name
                  ? <ChevronUp className="w-4 h-4 text-slate-400" />
                  : <ChevronDown className="w-4 h-4 text-slate-400" />
                }
              </div>
            </div>

            {expandedModel === model.model_name && (
              <div className="mt-4 pt-4 border-t border-surface-700 grid grid-cols-3 sm:grid-cols-5 gap-3 animate-fade-in">
                {isClassification ? (
                  <>
                    <MetricChip label="Accuracy"  value={(model.accuracy  * 100).toFixed(2) + '%'} />
                    <MetricChip label="Precision" value={(model.precision * 100).toFixed(2) + '%'} />
                    <MetricChip label="Recall"    value={(model.recall    * 100).toFixed(2) + '%'} />
                    <MetricChip label="F1 Score"  value={(model.f1_score  * 100).toFixed(2) + '%'} />
                    {model.roc_auc && <MetricChip label="ROC AUC" value={(model.roc_auc * 100).toFixed(2) + '%'} />}
                  </>
                ) : (
                  <>
                    <MetricChip label="R² Score" value={(model.r2 * 100).toFixed(2) + '%'} />
                    <MetricChip label="RMSE"     value={model.rmse.toFixed(4)} />
                    <MetricChip label="MAE"      value={model.mae.toFixed(4)} />
                  </>
                )}
                <MetricChip label="CV Mean" value={(model.cv_score_mean * 100).toFixed(2) + '%'} />
                <MetricChip label="CV Std"  value={(model.cv_score_std  * 100).toFixed(2) + '%'} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function MetricChip({ label, value }) {
  return (
    <div className="card-sm text-center">
      <p className="font-bold text-white text-sm">{value}</p>
      <p className="metric-label">{label}</p>
    </div>
  )
}

function FeaturesTab({ results }) {
  const fi = results.feature_importance
  if (!fi || Object.keys(fi).length === 0) {
    return <div className="text-center text-slate-400 py-12">No feature importance data available for this model.</div>
  }

  const sorted = Object.entries(fi).sort((a, b) => b[1] - a[1]).slice(0, 15)
  const maxVal = sorted[0][1]

  return (
    <div className="card animate-fade-in space-y-3">
      <h3 className="section-title">Feature Importance – {results.best_model}</h3>
      <div className="space-y-2">
        {sorted.map(([name, score], i) => (
          <div key={name} className="space-y-1">
            <div className="flex justify-between text-xs">
              <span className="font-mono text-slate-300 truncate max-w-[60%]">{name}</span>
              <span className="text-slate-400">{(score * 100).toFixed(3)}%</span>
            </div>
            <div className="h-2 bg-surface-700 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${(score / maxVal) * 100}%`,
                  background: `hsl(${240 - i * 12}, 70%, 65%)`,
                }}
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function VisualizationsTab({ results }) {
  const vizs = results.visualizations || {}
  const entries = Object.entries(vizs)

  if (entries.length === 0) {
    return <div className="text-center text-slate-400 py-12"><Image className="w-10 h-10 mx-auto mb-3 opacity-30" />No visualizations available.</div>
  }

  const labels = {
    correlation_heatmap:   'Correlation Heatmap',
    target_distribution:   'Target Distribution',
    missing_values:        'Missing Values',
    feature_distributions: 'Feature Distributions',
    outlier_boxplots:      'Outlier Box Plots',
    model_comparison:      'Model Comparison',
    feature_importance:    'Feature Importance',
    confusion_matrix:      'Confusion Matrix',
    roc_curve:             'ROC Curve',
    residuals:             'Regression Residuals',
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-fade-in">
      {entries.map(([key, url]) => (
        <div key={key} className="card !p-4">
          <p className="text-sm font-semibold text-slate-300 mb-3">{labels[key] || key}</p>
          <img
            src={url}
            alt={labels[key] || key}
            className="w-full rounded-lg border border-surface-700"
            loading="lazy"
            onError={e => { e.target.style.display = 'none' }}
          />
        </div>
      ))}
    </div>
  )
}
