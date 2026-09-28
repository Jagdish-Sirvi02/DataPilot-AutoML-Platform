import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Layout from './components/layout/Layout'
import UploadPage from './pages/UploadPage'
import PipelinePage from './pages/PipelinePage'
import ResultsPage from './pages/ResultsPage'
import EDAPage from './pages/EDAPage'

export default function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          style: { background: '#1e293b', color: '#e2e8f0', border: '1px solid #334155' },
          success: { iconTheme: { primary: '#10b981', secondary: '#0f172a' } },
          error:   { iconTheme: { primary: '#f43f5e', secondary: '#0f172a' } },
        }}
      />
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<UploadPage />} />
          <Route path="pipeline/:jobId" element={<PipelinePage />} />
          <Route path="results/:jobId" element={<ResultsPage />} />
          <Route path="eda/:datasetId" element={<EDAPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
