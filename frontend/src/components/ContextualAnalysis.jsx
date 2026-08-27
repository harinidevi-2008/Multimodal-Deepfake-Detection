// The backend only ever populates detected_mismatch when it identified a
// specific contextual mismatch (see evidence/evidence_builder.py's reasons
// and api/serializer.py's _build_contextual_analysis) - it's legitimately
// null when nothing specific stood out. That's not "no evidence available",
// it's a genuine "nothing specific was flagged" result, so it gets its own
// plain-language fallback rather than the app's usual "___ unavailable"
// empty state.
const NO_MISMATCH_MESSAGE = 'No specific contextual mismatch was identified.'

export default function ContextualAnalysis({ contextualAnalysis }) {
  const detectedMismatch = contextualAnalysis?.detected_mismatch
  const explanation = contextualAnalysis?.explanation

  return (
    <div className="card">
      <div className="card__header">
        <div>
          <div className="card__title">Contextual analysis</div>
          <div className="card__subtitle">Plain-language read on what stood out</div>
        </div>
      </div>
      <div className="stack" style={{ gap: 'var(--space-3)' }}>
        <div className="callout">
          <span className="callout-strong">Detected mismatch: </span>
          {detectedMismatch || NO_MISMATCH_MESSAGE}
        </div>
        {explanation && (
          <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            {explanation}
          </p>
        )}
      </div>
    </div>
  )
}
