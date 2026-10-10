const NO_MISMATCH_MESSAGE = 'No specific mismatch was identified.'

export default function ContextualAnalysis({ contextualAnalysis, transcript = '', segments = [] }) {
  const detectedMismatch = contextualAnalysis?.detected_mismatch || NO_MISMATCH_MESSAGE
  const summary = contextualAnalysis?.summary || contextualAnalysis?.explanation
  const findings = contextualAnalysis?.key_findings || []

  return (
    <div className="card">
      <div className="stack" style={{ gap: 'var(--space-3)' }}>
        {summary && (
          <p style={{ fontSize: 14, color: 'var(--text-primary)', lineHeight: 1.65, margin: 0 }}>
            {summary}
          </p>
        )}

        <div className="callout">
          <span className="callout-strong">Detected mismatch: </span>
          {detectedMismatch}
        </div>

        {findings.length > 0 && (
          <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.6 }}>
            {findings.slice(0, 4).map((item) => <li key={item}>{item}</li>)}
          </ul>
        )}
        <div>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Transcript</div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.6, margin: 0 }}>
            {transcript || 'No transcript was returned.'}
          </p>
          {segments.length > 0 && <ul style={{ margin: '8px 0 0', paddingLeft: 18, fontSize: 12, color: 'var(--text-muted)' }}>
            {segments.map((segment, index) => <li key={`${segment.start}-${index}`}>{segment.start != null ? `${Number(segment.start).toFixed(1)}s: ` : ''}{segment.text}</li>)}
          </ul>}
        </div>
      </div>
    </div>
  )
}
