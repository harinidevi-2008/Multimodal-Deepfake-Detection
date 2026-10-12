import { toPercent } from '../utils/formatters'

export default function OverallResultPanel({ result }) {
  const isFake = result.final_verdict === 'LIKELY_DEEPFAKE'
  const fakeLikelihood = result.overall_fake_likelihood ?? result.final_fake_probability
  const hasScore = typeof fakeLikelihood === 'number' && Number.isFinite(fakeLikelihood)

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
            {hasScore ? toPercent(fakeLikelihood, 0) : 'Unavailable'}
          </h2>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
            {result.video_filename || 'Uploaded video'}
          </span>
        </div>

        <div>
          <div style={{ fontSize: 12.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Final fake likelihood
          </div>
          <div style={{ fontSize: 42, lineHeight: 1.1, fontWeight: 800, color: isFake ? 'var(--danger-strong)' : 'var(--success-strong)' }}>
            {hasScore ? toPercent(fakeLikelihood, 0) : 'Unavailable'}
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', margin: '8px 0 0' }}>
            {hasScore
              ? 'This is the combined score from the visual, audio, and semantic analyses. It is a model score, not proof of a specific altered moment.'
              : 'A validated three-stream fusion prediction is unavailable for this analysis.'}
          </p>
        </div>
      </div>
    </div>
  )
}
