// Shown in place of a chart/list when the backend genuinely didn't produce
// that evidence for this request (e.g. lip-sync window evidence requires
// the raw video and a successful analyze_lipsync(..., compute_windows=True)
// pass). Never a substitute for missing data with a fabricated chart.
export default function EmptyState({ title, body }) {
  return (
    <div
      className="callout"
      style={{ textAlign: 'center', padding: 'var(--space-6) var(--space-5)' }}
    >
      <p className="callout-strong" style={{ marginBottom: body ? 6 : 0 }}>
        {title}
      </p>
      {body && <p>{body}</p>}
    </div>
  )
}
