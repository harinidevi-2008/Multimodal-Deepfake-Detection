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
  const semantic = contextualAnalysis?.semantic_context

  return (
    <div className="card">
      <div className="card__header">
        <div>
          <div className="card__title">Contextual analysis</div>
          <div className="card__subtitle">Plain-language read on what stood out</div>
        </div>
      </div>
      <div className="stack" style={{ gap: 'var(--space-3)' }}>
        {contextualAnalysis?.summary && (
          <p style={{ fontSize: 14, color: 'var(--text-primary)', lineHeight: 1.65, margin: 0 }}>
            {contextualAnalysis.summary}
          </p>
        )}
        <div className="callout">
          <span className="callout-strong">Detected mismatch: </span>
          {detectedMismatch || NO_MISMATCH_MESSAGE}
        </div>
        <AnalysisBlock title="Key findings" items={contextualAnalysis?.key_findings} />
        <SignalBlock title="Supporting final verdict" signals={contextualAnalysis?.supporting_signals} />
        <SignalBlock title="Contradicting final verdict" signals={contextualAnalysis?.contradicting_signals} />
        {contextualAnalysis?.cross_modal_interpretation && (
          <TextBlock title="Cross-modality interpretation" text={contextualAnalysis.cross_modal_interpretation} />
        )}
        {contextualAnalysis?.modality_explanations?.length > 0 && (
          <ModalityExplanations items={contextualAnalysis.modality_explanations} />
        )}
        {contextualAnalysis?.fusion_interpretation && (
          <TextBlock title="Fusion interpretation" text={contextualAnalysis.fusion_interpretation} />
        )}
        <AnalysisBlock title="Evidence limitations" items={contextualAnalysis?.evidence_limitations} />
        {semantic && (
          <div className="callout">
            <span className="callout-strong">Semantic context: </span>
            {semantic.explanation}
              <div style={{ marginTop: 6 }}>Language: {semantic.language || 'unknown'}</div>
            {semantic.transcript && <div style={{ marginTop: 6 }}>Transcript: {semantic.transcript}</div>}
            {semantic.segments?.length > 0 && (
              <div style={{ marginTop: 6 }}>
                Segments: {semantic.segments.map((segment) => `${segment.start.toFixed(1)}-${segment.end.toFixed(1)}s`).join(', ')}
              </div>
            )}
          </div>
        )}
        {contextualAnalysis?.overall_interpretation && (
          <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            {contextualAnalysis.overall_interpretation}
          </p>
        )}
        {!contextualAnalysis?.summary && explanation && (
          <p style={{ fontSize: 13.5, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
            {explanation}
          </p>
        )}
      </div>
    </div>
  )
}

function ModalityExplanations({ items }) {
  return (
    <div>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 6 }}>Modality-by-modality explanation</div>
      <div className="stack" style={{ gap: 'var(--space-2)' }}>
        {items.map((item) => (
          <div key={item.modality} className="callout">
            <div style={{ fontWeight: 700, marginBottom: 4 }}>
              {item.label} — {item.supports === 'FAKE' ? 'supports FAKE' : 'supports REAL'} ({item.strength})
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
              <div><strong>What it analyzes:</strong> {item.what_it_analyzes}</div>
              <div><strong>Signal meaning:</strong> {item.signal_meaning}</div>
              <div><strong>Limitation:</strong> {item.limitation}</div>
              <div style={{ marginTop: 4 }}><strong>Interpretation:</strong> {item.explanation}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function AnalysisBlock({ title, items }) {
  if (!items?.length) return null
  return (
    <div>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{title}</div>
      <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.6 }}>
        {items.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </div>
  )
}

function SignalBlock({ title, signals }) {
  if (!signals?.length) return null
  return (
    <div>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{title}</div>
      <ul style={{ margin: 0, paddingLeft: 18, color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.6 }}>
        {signals.map((signal) => (
          <li key={`${signal.modality}-${signal.reason}`}>{signal.reason}</li>
        ))}
      </ul>
    </div>
  )
}

function TextBlock({ title, text }) {
  return (
    <div>
      <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{title}</div>
      <p style={{ margin: 0, color: 'var(--text-secondary)', fontSize: 13, lineHeight: 1.6 }}>{text}</p>
    </div>
  )
}
