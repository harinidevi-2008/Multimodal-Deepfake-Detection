import { useEffect, useRef, useState } from 'react'
import { ANALYSIS_STAGES, analyzeVideo } from '../api/analyzeVideo'

export default function AnalysisScreen({ file, onComplete, onError }) {
  const [jobStatus, setJobStatus] = useState({
    status: 'queued', stage: 'queued', message: 'Queued for analysis.'
  })
  const previewUrl = useRef(file ? URL.createObjectURL(file) : null)
  const started = useRef(false)

  useEffect(() => {
    if (started.current) return
    started.current = true
    analyzeVideo(file, setJobStatus)
      .then(onComplete)
      .catch((err) => onError?.(err))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const activeIndex = ANALYSIS_STAGES.findIndex((stage) => stage.key === jobStatus.stage)

  return (
    <div className="stack" style={{ gap: 'var(--space-6)', maxWidth: 680, margin: '0 auto' }}>
      <div className="stack" style={{ gap: 'var(--space-2)', textAlign: 'center' }}>
        <h1 style={{ fontSize: 24 }}>Analyzing your video...</h1>
        <p style={{ color: 'var(--text-secondary)' }}>{jobStatus.message}</p>
      </div>

      <div className="card">
        <div className="grid grid-cols-2" style={{ gap: 'var(--space-5)', alignItems: 'center' }}>
          <div style={{ position: 'relative' }}>
            {previewUrl.current ? (
              <video
                src={previewUrl.current}
                muted
                autoPlay
                loop
                style={{ width: '100%', borderRadius: 'var(--radius-md)', maxHeight: 220, background: '#000' }}
              />
            ) : (
              <div
                style={{
                  width: '100%',
                  height: 180,
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-subtle)'
                }}
              />
            )}
            <ScanOverlay />
          </div>

          <ol className="stack" style={{ gap: 'var(--space-3)', listStyle: 'none', margin: 0, padding: 0 }}>
            {ANALYSIS_STAGES.map((stage, i) => {
              const isDone = activeIndex >= 0 && i < activeIndex
              const isActive = i === activeIndex
              return (
                <li key={stage.key} className="stack" style={{ flexDirection: 'row', alignItems: 'center', gap: 'var(--space-3)' }}>
                  <StageIcon state={isDone ? 'done' : isActive ? 'active' : 'pending'} />
                  <span
                    style={{
                      fontSize: 13.5,
                      fontWeight: isActive ? 700 : 600,
                      color: isDone
                        ? 'var(--text-primary)'
                        : isActive
                          ? 'var(--accent)'
                          : 'var(--text-muted)'
                    }}
                  >
                    {stage.label}
                  </span>
                </li>
              )
            })}
          </ol>
        </div>
      </div>
    </div>
  )
}

function StageIcon({ state }) {
  if (state === 'done') {
    return (
      <span
        style={{
          width: 22,
          height: 22,
          borderRadius: 999,
          background: 'var(--success)',
          color: 'white',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontSize: 12,
          flexShrink: 0
        }}
      >
        OK
      </span>
    )
  }
  if (state === 'active') {
    return (
      <span
        style={{
          width: 22,
          height: 22,
          borderRadius: 999,
          border: '2px solid var(--accent)',
          borderTopColor: 'transparent',
          flexShrink: 0,
          animation: 'spin 0.8s linear infinite'
        }}
      />
    )
  }
  return (
    <span
      style={{
        width: 22,
        height: 22,
        borderRadius: 999,
        border: '2px solid var(--border-strong)',
        flexShrink: 0
      }}
    />
  )
}

function ScanOverlay() {
  return (
    <div
      aria-hidden
      style={{
        position: 'absolute',
        inset: 0,
        borderRadius: 'var(--radius-md)',
        overflow: 'hidden',
        pointerEvents: 'none'
      }}
    >
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          height: '30%',
          background:
            'linear-gradient(180deg, rgba(52,87,213,0) 0%, rgba(52,87,213,0.28) 50%, rgba(52,87,213,0) 100%)',
          animation: 'scan 2.2s ease-in-out infinite'
        }}
      />
    </div>
  )
}
