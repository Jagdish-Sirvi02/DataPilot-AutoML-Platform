import axios from 'axios'

const BASE = '/api'

const api = axios.create({
  baseURL: BASE,
  timeout: 30_000,
})

// ── Dataset ────────────────────────────────────────────────────────────────

export async function uploadDataset(file, onProgress) {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post('/upload-dataset', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => onProgress && onProgress(Math.round((e.loaded * 100) / e.total)),
  })
  return data
}

export async function getDatasetSummary(datasetId) {
  const { data } = await api.get(`/dataset-summary/${datasetId}`)
  return data
}

// ── AutoML ─────────────────────────────────────────────────────────────────

export async function runAutoML(payload) {
  const { data } = await api.post('/run-automl', payload)
  return data   // { job_id, message, status }
}

export async function getJobStatus(jobId) {
  const { data } = await api.get(`/job-status/${jobId}`)
  return data
}

// ── Results ────────────────────────────────────────────────────────────────

export async function getModelResults(jobId) {
  const { data } = await api.get(`/model-results/${jobId}`)
  return data
}

export async function getVisualizations(jobId) {
  const { data } = await api.get(`/visualizations/${jobId}`)
  return data
}

// ── Workflow ───────────────────────────────────────────────────────────────

export async function getWorkflowInfo(jobId) {
  const { data } = await api.get(`/workflow-info/${jobId}`)
  return data
}

export function downloadWorkflowUrl(jobId) {
  return `${BASE}/download-workflow/${jobId}`
}

// ── Polling helper ─────────────────────────────────────────────────────────

/**
 * Poll job status until complete or failed.
 * @param {string} jobId
 * @param {function} onUpdate - called with status object each tick
 * @param {number} intervalMs
 * @returns cleanup function
 */
export function pollJobStatus(jobId, onUpdate, intervalMs = 1500) {
  let active = true
  const tick = async () => {
    if (!active) return
    try {
      const status = await getJobStatus(jobId)
      onUpdate(status)
      if (status.status !== 'completed' && status.status !== 'failed') {
        setTimeout(tick, intervalMs)
      }
    } catch (err) {
      console.error('Poll error:', err)
      setTimeout(tick, intervalMs * 2)
    }
  }
  tick()
  return () => { active = false }
}
