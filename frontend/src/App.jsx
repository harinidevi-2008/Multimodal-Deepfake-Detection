import { useState } from 'react'
import UploadScreen from './screens/UploadScreen'
import AnalysisScreen from './screens/AnalysisScreen'
import ResultsDashboard from './screens/ResultsDashboard'
import ErrorScreen from './screens/ErrorScreen'

// Simple state machine: upload -> analyzing -> results (or error).
// See src/api/analyzeVideo.js for the network call and API_CONTRACT.md for
// the response shape ResultsDashboard and its children read from.
const STEPS = {
  UPLOAD: 'upload',
  ANALYZING: 'analyzing',
  RESULTS: 'results',
  ERROR: 'error'
}

export default function App() {
  const [step, setStep] = useState(STEPS.UPLOAD)
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleAnalyze = (selectedFile) => {
    setFile(selectedFile)
    setError(null)
    setStep(STEPS.ANALYZING)
  }

  const handleComplete = (analysisResult) => {
    setResult(analysisResult)
    setStep(STEPS.RESULTS)
  }

  const handleError = (err) => {
    // Full error (including any details from the backend) is logged for
    // debugging; only the structured, user-safe fields render in the UI.
    console.error('Analysis failed:', err)
    setError(err)
    setStep(STEPS.ERROR)
  }

  const handleReset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setStep(STEPS.UPLOAD)
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__inner">
          <div className="brand">
            <span className="brand__mark">TF</span>
            Truthframe
            <span className="brand__suffix">Deepfake Detection</span>
          </div>
          <StepIndicator step={step} />
        </div>
      </header>

      <main className="app-main">
        {step === STEPS.UPLOAD && <UploadScreen onAnalyze={handleAnalyze} />}
        {step === STEPS.ANALYZING && (
          <AnalysisScreen file={file} onComplete={handleComplete} onError={handleError} />
        )}
        {step === STEPS.RESULTS && result && (
          <ResultsDashboard result={result} onReset={handleReset} />
        )}
        {step === STEPS.ERROR && <ErrorScreen error={error} onRetry={handleReset} />}
      </main>

      <footer className="app-footer">
        {import.meta.env.VITE_USE_MOCK_API === 'true'
          ? 'Running against bundled mock JSON (VITE_USE_MOCK_API=true).'
          : 'Calling the backend at ' +
            (import.meta.env.VITE_API_BASE_URL || '/api (via dev proxy)') +
            ' — see API_CONTRACT.md.'}
      </footer>
    </div>
  )
}

function StepIndicator({ step }) {
  const steps = [
    { key: STEPS.UPLOAD, label: 'Upload' },
    { key: STEPS.ANALYZING, label: 'Analyze' },
    { key: STEPS.RESULTS, label: 'Results' }
  ]
  // Error can happen mid-analysis; treat it as still "on" the Analyze step
  // for the purposes of the indicator rather than inventing a 4th dot.
  const effectiveStep = step === STEPS.ERROR ? STEPS.ANALYZING : step
  const activeIndex = steps.findIndex((s) => s.key === effectiveStep)

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      {steps.map((s, i) => (
        <div key={s.key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span
            style={{
              fontSize: 12.5,
              fontWeight: 700,
              color: i <= activeIndex ? 'var(--accent)' : 'var(--text-muted)'
            }}
          >
            {s.label}
          </span>
          {i < steps.length - 1 && (
            <span style={{ width: 20, height: 1, background: 'var(--border)' }} />
          )}
        </div>
      ))}
    </div>
  )
}
