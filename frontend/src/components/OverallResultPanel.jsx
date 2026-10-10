import { toPercent } from '../utils/formatters'

export default function OverallResultPanel({ result }) {
  const value = result.overall_fake_likelihood
  const fakePct = Number.isFinite(Number(value)) ? Number(value) : null

  return (
    <div
      className="card"
      style={{
        background: 'linear-gradient(180deg, var(--accent-soft) 0%, var(--surface) 55%)',
        borderColor: 'var(--border)'
      }}
    >
      <div className="stack" style={{ gap: 'var(--space-3)' }}>
        <span style={{ fontSize: 12.5, fontWeight: 700, letterSpacing: '0.04em', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
          Overall Fake Likelihood
        </span>

        <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
          <h2 style={{ fontSize: 42, margin: 0, color: 'var(--accent-strong)' }}>
            {fakePct === null ? 'Not available' : toPercent(fakePct, 0)}
          </h2>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            {result.video_filename || 'Uploaded video'}
          </span>
        </div>

        <div>
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>
          Reliability-weighted mean of available learned modality scores.
          </div>
        </div>
      </div>
    </div>
  )
}
