import Badge from './ui/Badge'
import EmptyState from './ui/EmptyState'
import { toPercent, scoreTone } from '../utils/formatters'

// Backend status vocabulary isn't fully pinned down yet (blink status has
// been described as both "Irregular" and "Abnormal" across drafts) — treat
// any of the known "flagged" values as warning-toned, everything else as
// the calm/consistent tone.
const FLAGGED_STATUSES = new Set(['Irregular', 'Abnormal', 'Inconsistent'])
function isFlaggedStatus(status) {
  return FLAGGED_STATUSES.has(status)
}

export default function ModalityCard({ meta, data }) {
  const isRuleBased = meta.kind === 'rule_based_score'
  // `data` can genuinely be missing/null (a partial result), or present but
  // without the score this modality's "kind" expects - never fabricate a
  // value in either case, just show the same clean unavailable state a
  // missing evidence tab already uses elsewhere in this app.
  const value = data ? (isRuleBased ? data.anomaly_score ?? data.mismatch_score : data.fake_probability) : undefined

  if (value === undefined || value === null) {
    return (
      <div className="card card--tight stack" style={{ gap: 'var(--space-3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 9, height: 9, borderRadius: 999, background: meta.color }} />
          <span style={{ fontWeight: 700, fontSize: 13.5 }}>{meta.label}</span>
        </div>
        <EmptyState title="Analysis unavailable" body="No result was returned for this modality." />
      </div>
    )
  }

  const tone = scoreTone(value)

  return (
    <div className="card card--tight stack" style={{ gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 9, height: 9, borderRadius: 999, background: meta.color }} />
          <span style={{ fontWeight: 700, fontSize: 13.5 }}>{meta.label}</span>
        </div>
        {isRuleBased ? (
          <Badge tone="neutral">rule-based</Badge>
        ) : (
          <Badge tone="accent">learned</Badge>
        )}
      </div>

      <div>
        <div className="mono" style={{ fontSize: 26, fontWeight: 800, color: `var(--${tone})` }}>
          {toPercent(value, 0)}
        </div>
        <div style={{ fontSize: 11.5, color: 'var(--text-muted)', fontWeight: 600 }}>
          {meta.metricLabel}
        </div>
      </div>

      <div className="meter">
        <div className="meter__fill" style={{ width: toPercent(value, 0), background: meta.color }} />
      </div>

      {isRuleBased && data.status && (
        <Badge tone={isFlaggedStatus(data.status) ? 'warning' : 'success'}>{data.status}</Badge>
      )}

      <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
        {meta.description}
      </p>
    </div>
  )
}
