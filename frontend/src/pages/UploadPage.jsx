import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import toast from 'react-hot-toast'
import {
  Upload, FileText, CheckCircle2, AlertCircle,
  ChevronDown, Sparkles, Database, Table2
} from 'lucide-react'
import clsx from 'clsx'
import { uploadDataset, runAutoML } from '../services/api'

const PROBLEM_TYPES = [
  { value: '', label: 'Auto-detect', desc: 'Let DataPilot decide' },
  { value: 'classification', label: 'Classification', desc: 'Predict a category' },
  { value: 'regression', label: 'Regression', desc: 'Predict a number' },
]

export default function UploadPage() {
  const navigate = useNavigate()
  const [file, setFile]           = useState(null)
  const [metadata, setMetadata]   = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadPct, setUploadPct] = useState(0)
  const [targetCol, setTargetCol] = useState('')
  const [problemType, setProblemType] = useState('')
  const [excludeCols, setExcludeCols] = useState([])
  const [launching, setLaunching] = useState(false)

  const onDrop = useCallback(async (accepted) => {
    const f = accepted[0]
    if (!f) return
    if (!f.name.endsWith('.csv')) {
      toast.error('Only CSV files are supported')
      return
    }
    setFile(f)
    setMetadata(null)
    setUploading(true)
    setUploadPct(0)

    try {
      const meta = await uploadDataset(f, setUploadPct)
      setMetadata(meta)
      setTargetCol(meta.target_column || '')
      toast.success(`Dataset uploaded: ${meta.rows.toLocaleString()} rows × ${meta.columns} columns`)
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Upload failed')
      setFile(null)
    } finally {
      setUploading(false)
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'] }, maxFiles: 1,
  })

  const toggleExclude = (col) => {
    setExcludeCols(prev =>
      prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]
    )
  }

  const handleLaunch = async () => {
    if (!metadata || !targetCol) {
      toast.error('Please select a target column')
      return
    }
    if (excludeCols.includes(targetCol)) {
      toast.error('Cannot exclude the target column')
      return
    }
    setLaunching(true)
    try {
      const { job_id } = await runAutoML({
        dataset_id: metadata.dataset_id,
        target_column: targetCol,
        problem_type: problemType || null,
        exclude_columns: excludeCols,
      })
      toast.success('AutoML pipeline launched!')
      navigate(`/pipeline/${job_id}`)
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Failed to start pipeline')
    } finally {
      setLaunching(false)
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in">
      {/* Hero */}
      <div className="text-center py-4">
        <h2 className="text-3xl font-bold gradient-text mb-2">Upload Your Dataset</h2>
        <p className="text-slate-400">
          Drop a CSV file and DataPilot will automatically train, evaluate, and export your ML pipeline.
        </p>
      </div>

      {/* Drop zone */}
      <div
        {...getRootProps()}
        className={clsx(
          'card border-2 border-dashed cursor-pointer transition-all duration-200 text-center py-12',
          isDragActive
            ? 'border-brand-400 bg-brand-500/10'
            : metadata
            ? 'border-emerald-500/50 bg-emerald-500/5'
            : 'border-surface-600 hover:border-brand-500/60 hover:bg-brand-500/5'
        )}
      >
        <input {...getInputProps()} />
        {metadata ? (
          <div className="space-y-2">
            <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto" />
            <p className="font-semibold text-emerald-300 text-lg">{file.name}</p>
            <p className="text-slate-400 text-sm">
              {metadata.rows.toLocaleString()} rows · {metadata.columns} columns · {metadata.file_size_kb} KB
            </p>
            <p className="text-xs text-slate-500">Click or drop to replace</p>
          </div>
        ) : uploading ? (
          <div className="space-y-4">
            <div className="w-12 h-12 border-4 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto" />
            <p className="text-slate-300 font-medium">Uploading… {uploadPct}%</p>
            <div className="w-48 h-1.5 bg-surface-700 rounded-full mx-auto overflow-hidden">
              <div className="h-full bg-brand-500 rounded-full progress-bar-fill" style={{ width: `${uploadPct}%` }} />
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <Upload className={clsx('w-12 h-12 mx-auto', isDragActive ? 'text-brand-400' : 'text-slate-500')} />
            <div>
              <p className="font-semibold text-slate-200">
                {isDragActive ? 'Drop your CSV here' : 'Drag & drop your CSV file'}
              </p>
              <p className="text-slate-500 text-sm mt-1">or click to browse · Max 100 MB</p>
            </div>
          </div>
        )}
      </div>

      {/* Dataset info + config */}
      {metadata && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-slide-up">
          {/* Dataset stats */}
          <div className="card space-y-4">
            <h3 className="section-title flex items-center gap-2">
              <Database className="w-4 h-4 text-brand-400" />
              Dataset Overview
            </h3>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'Rows', value: metadata.rows.toLocaleString() },
                { label: 'Columns', value: metadata.columns },
                { label: 'Duplicates', value: metadata.duplicate_rows },
                { label: 'Has Missing', value: metadata.has_missing ? 'Yes' : 'No' },
                { label: 'Numeric Cols', value: metadata.numeric_columns.length },
                { label: 'Categorical', value: metadata.categorical_columns.length },
              ].map(({ label, value }) => (
                <div key={label} className="card-sm !p-3">
                  <p className="metric-value text-xl">{value}</p>
                  <p className="metric-label">{label}</p>
                </div>
              ))}
            </div>
            {metadata.has_missing && (
              <div className="flex items-start gap-2 text-amber-400 text-xs bg-amber-500/10 rounded-lg p-3 border border-amber-500/20">
                <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                <span>Missing values detected. DataPilot will handle them automatically.</span>
              </div>
            )}
          </div>

          {/* Configuration */}
          <div className="card space-y-5">
            <h3 className="section-title flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-brand-400" />
              Pipeline Configuration
            </h3>

            {/* Target column */}
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Target Column *
              </label>
              <div className="relative">
                <select
                  value={targetCol}
                  onChange={e => setTargetCol(e.target.value)}
                  className="input appearance-none pr-10"
                >
                  <option value="">Select target column…</option>
                  {metadata.column_info.map(c => (
                    <option key={c.name} value={c.name}>
                      {c.name} ({c.column_type})
                    </option>
                  ))}
                </select>
                <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400 pointer-events-none" />
              </div>
              {metadata.target_column && (
                <p className="text-xs text-slate-500 mt-1.5">
                  Auto-detected: <span className="text-brand-400 font-mono">{metadata.target_column}</span>
                  {metadata.problem_type && (
                    <> · <span className="text-emerald-400">{metadata.problem_type}</span></>
                  )}
                </p>
              )}
            </div>

            {/* Problem type */}
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                Problem Type
              </label>
              <div className="grid grid-cols-3 gap-2">
                {PROBLEM_TYPES.map(pt => (
                  <button
                    key={pt.value}
                    onClick={() => setProblemType(pt.value)}
                    className={clsx(
                      'text-left p-3 rounded-xl border text-xs transition-all duration-150',
                      problemType === pt.value
                        ? 'border-brand-500 bg-brand-500/15 text-brand-300'
                        : 'border-surface-600 bg-surface-700 text-slate-400 hover:border-surface-500'
                    )}
                  >
                    <div className="font-semibold mb-0.5">{pt.label}</div>
                    <div className="text-slate-500 text-[10px]">{pt.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Column exclusion */}
      {metadata && (
        <div className="card animate-slide-up">
          <h3 className="section-title flex items-center gap-2 mb-4">
            <Table2 className="w-4 h-4 text-brand-400" />
            Column Preview
            <span className="badge badge-slate ml-auto">{metadata.columns} columns</span>
            {excludeCols.length > 0 && (
              <span className="badge badge-red">{excludeCols.length} excluded</span>
            )}
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-surface-700">
                  {['Column', 'Type', 'Missing %', 'Unique', 'Sample Values', 'Exclude'].map(h => (
                    <th key={h} className="text-left py-2 px-3 text-xs font-semibold text-slate-400 uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metadata.column_info.map((col) => (
                  <tr
                    key={col.name}
                    className={clsx(
                      'border-b border-surface-700/50 transition-colors',
                      col.name === targetCol
                        ? 'bg-brand-500/10'
                        : excludeCols.includes(col.name)
                        ? 'bg-rose-500/5 opacity-50'
                        : 'hover:bg-surface-700/40'
                    )}
                  >
                    <td className="py-2.5 px-3 font-mono text-xs font-medium text-slate-200">
                      {col.name}
                      {col.name === targetCol && (
                        <span className="ml-2 badge badge-blue text-[10px]">target</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3">
                      <span className={clsx('badge text-[10px]',
                        col.column_type === 'numeric' ? 'badge-blue'
                        : col.column_type === 'categorical' ? 'badge-yellow'
                        : 'badge-slate'
                      )}>
                        {col.column_type}
                      </span>
                    </td>
                    <td className={clsx(
                      'py-2.5 px-3 text-xs',
                      col.missing_pct > 20 ? 'text-rose-400' : col.missing_pct > 0 ? 'text-amber-400' : 'text-slate-400'
                    )}>
                      {col.missing_pct}%
                    </td>
                    <td className="py-2.5 px-3 text-xs text-slate-400">{col.unique_count.toLocaleString()}</td>
                    <td className="py-2.5 px-3 text-xs text-slate-500 font-mono truncate max-w-[180px]">
                      {col.sample_values.slice(0, 3).join(', ')}
                    </td>
                    <td className="py-2.5 px-3">
                      {col.name !== targetCol && (
                        <button
                          onClick={() => toggleExclude(col.name)}
                          className={clsx(
                            'w-5 h-5 rounded border-2 transition-all',
                            excludeCols.includes(col.name)
                              ? 'bg-rose-500 border-rose-400'
                              : 'bg-transparent border-surface-500 hover:border-rose-400'
                          )}
                        />
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Launch button */}
      {metadata && (
        <div className="flex justify-end animate-fade-in">
          <button
            onClick={handleLaunch}
            disabled={!targetCol || launching}
            className="btn-primary flex items-center gap-2 text-base px-8 py-3"
          >
            {launching ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Launching…
              </>
            ) : (
              <>
                <Sparkles className="w-4 h-4" />
                Launch AutoML Pipeline
              </>
            )}
          </button>
        </div>
      )}
    </div>
  )
}
