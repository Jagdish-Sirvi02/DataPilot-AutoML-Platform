import { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, BarChart2, AlertCircle } from 'lucide-react'
import { getDatasetSummary } from '../services/api'

export default function EDAPage() {
  const { datasetId } = useParams()
  const navigate = useNavigate()
  const [meta, setMeta] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getDatasetSummary(datasetId)
      .then(setMeta)
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [datasetId])

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <div className="w-10 h-10 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
    </div>
  )

  if (!meta) return (
    <div className="text-center py-20">
      <AlertCircle className="w-10 h-10 mx-auto mb-3 text-rose-400" />
      <p className="text-slate-400">Dataset not found.</p>
      <button onClick={() => navigate('/')} className="btn-secondary mt-4">← Back</button>
    </div>
  )

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      <div>
        <button onClick={() => navigate('/')} className="btn-ghost flex items-center gap-1.5 text-sm mb-3">
          <ArrowLeft className="w-3.5 h-3.5" /> Back
        </button>
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <BarChart2 className="w-6 h-6 text-brand-400" />
          Dataset Explorer
        </h2>
        <p className="text-slate-400 text-sm mt-1 font-mono">{meta.filename}</p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: 'Rows',        value: meta.rows.toLocaleString() },
          { label: 'Columns',     value: meta.columns },
          { label: 'Duplicates',  value: meta.duplicate_rows },
          { label: 'Size (KB)',   value: meta.file_size_kb },
        ].map(s => (
          <div key={s.label} className="card text-center">
            <p className="metric-value">{s.value}</p>
            <p className="metric-label">{s.label}</p>
          </div>
        ))}
      </div>

      <div className="card">
        <h3 className="section-title">Column Details</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-surface-700">
                {['Column', 'Type', 'Dtype', 'Missing %', 'Unique', 'Samples'].map(h => (
                  <th key={h} className="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {meta.column_info.map(col => (
                <tr key={col.name} className="border-b border-surface-700/50 hover:bg-surface-700/30">
                  <td className="py-2.5 px-3 font-mono text-xs text-slate-200">{col.name}</td>
                  <td className="py-2.5 px-3">
                    <span className={`badge text-[10px] ${
                      col.column_type === 'numeric' ? 'badge-blue'
                      : col.column_type === 'categorical' ? 'badge-yellow'
                      : 'badge-slate'
                    }`}>{col.column_type}</span>
                  </td>
                  <td className="py-2.5 px-3 text-xs text-slate-500 font-mono">{col.dtype}</td>
                  <td className={`py-2.5 px-3 text-xs ${col.missing_pct > 20 ? 'text-rose-400' : col.missing_pct > 0 ? 'text-amber-400' : 'text-slate-400'}`}>
                    {col.missing_pct}%
                  </td>
                  <td className="py-2.5 px-3 text-xs text-slate-400">{col.unique_count.toLocaleString()}</td>
                  <td className="py-2.5 px-3 text-xs text-slate-500 font-mono truncate max-w-[160px]">
                    {col.sample_values.slice(0, 3).join(', ')}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
