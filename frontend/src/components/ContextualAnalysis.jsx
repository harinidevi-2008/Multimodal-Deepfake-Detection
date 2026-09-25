const NO_MISMATCH_MESSAGE = 'No specific mismatch was identified.'

export default function ContextualAnalysis({ contextualAnalysis }) {
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
      </div>
    </div>
  )
}
