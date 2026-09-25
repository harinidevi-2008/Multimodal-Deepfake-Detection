import { toPercent } from '../utils/formatters'

export default function OverallResultPanel({ result }) {
  const isFake = result.final_verdict === 'LIKELY_DEEPFAKE'
  const fakePct = Number.isFinite(Number(result.final_fake_probability))
    ? Number(result.final_fake_probability)
    : 0

  return (
    <div
      className="card"
      style={{
        background: isFake
          ? 'linear-gradient(180deg, var(--danger-soft) 0%, var(--surface) 55%)'
          : 'linear-gradient(180deg, var(--success-soft) 0%, var(--surface) 55%)',
        borderColor: isFake ? 'var(--danger-soft-border)' : 'var(--success-soft-border)'
      }}
    >
      <div className="stack" style={{ gap: 'var(--space-3)' }}>
        <span style={{ fontSize: 12.5, fontWeight: 700, letterSpacing: '0.04em', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
          Final Result
        </span>

        <div style={{ display: 'flex', alignItems: 'baseline', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
          <h2 style={{ fontSize: 30, margin: 0, color: isFake ? 'var(--danger-strong)' : 'var(--success-strong)' }}>
            {isFake ? 'FAKE' : 'REAL'}
          </h2>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            {result.video_filename || 'Uploaded video'}
          </span>
        </div>

        <div>
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Overall fake likelihood
          </div>
          <div style={{ fontSize: 42, lineHeight: 1.1, fontWeight: 800, color: isFake ? 'var(--danger-strong)' : 'var(--success-strong)' }}>
            {toPercent(fakePct, 0)}
          </div>
        </div>
      </div>
    </div>
  )
}
