import Badge from '../components/ui/Badge'

// Shown when analyzeVideo() rejects — a real backend/network/validation
// failure, never a fabricated result. No stack traces surface here; the
// full error is logged to the console for debugging.
export default function ErrorScreen({ error, onRetry }) {
  const code = error?.code || 'unknown_error'
  const message = error?.message || 'Something went wrong while analyzing this video.'

  return (
    <div className="stack" style={{ gap: 'var(--space-5)', maxWidth: 560, margin: '0 auto', textAlign: 'center' }}>
      <div
        style={{
          width: 56,
          height: 56,
          borderRadius: 16,
          background: 'var(--danger-soft)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: '0 auto'
        }}
      >
        <ErrorGlyph />
      </div>

      <div className="stack" style={{ gap: 'var(--space-2)' }}>
        <h1 style={{ fontSize: 22 }}>Analysis couldn't be completed</h1>
        <p style={{ color: 'var(--text-secondary)' }}>{message}</p>
      </div>

      <div className="card card--tight" style={{ textAlign: 'left', boxShadow: 'none' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: 12.5, color: 'var(--text-muted)', fontWeight: 600 }}>Error code</span>
          <Badge tone="danger">{code}</Badge>
        </div>
        {error?.status != null && (
          <div className="kv-row">
            <span className="kv-row__key">HTTP status</span>
            <span className="kv-row__value">{error.status}</span>
          </div>
        )}
        {error?.details && (
          <div className="kv-row" style={{ alignItems: 'flex-start' }}>
            <span className="kv-row__key">Details</span>
            <span className="kv-row__value" style={{ textAlign: 'right', fontWeight: 500 }}>
              {String(error.details)}
            </span>
          </div>
        )}
      </div>

      <button className="btn btn-primary" style={{ margin: '0 auto' }} onClick={onRetry}>
        Try again
      </button>
    </div>
  )
}

function ErrorGlyph() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
      <path
        d="M12 9v4m0 4h.01M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z"
        stroke="var(--danger)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
