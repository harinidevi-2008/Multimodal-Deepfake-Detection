import Badge from './ui/Badge'
import EmptyState from './ui/EmptyState'
import { toPercent, scoreTone } from '../utils/formatters'

const FLAGGED_STATUSES = new Set(['Irregular', 'Abnormal', 'Inconsistent'])
function isFlaggedStatus(status) {
  return FLAGGED_STATUSES.has(status)
}

export default function ModalityCard({ meta, data, supplementaryEvidence }) {
  const isRuleBased = meta.kind === 'rule_based_score'
  const value = data ? (isRuleBased ? data.anomaly_score ?? data.mismatch_score : data.fake_probability) : undefined

  if (value === undefined || value === null) {
    return (
      <div className="card card--tight stack" style={{ gap: 'var(--space-3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 9, height: 9, borderRadius: 999, background: meta.color }} />
          <span style={{ fontWeight: 700, fontSize: 13.5 }}>{meta.label}</span>
        </div>
        <EmptyState title="Not available" body={data?.explanation || data?.reason || 'No result was returned for this modality.'} />
      </div>
    )
  }

  const tone = scoreTone(value)
  const statusLabel = data?.status
    ? (data.status === 'Unavailable' ? 'Unavailable' : (data.status === 'Abnormal' || data.status === 'Irregular' || data.status === 'Inconsistent' ? 'Anomalous' : 'Consistent'))
    : null

  return (
    <div className="card card--tight stack" style={{ gap: 'var(--space-3)' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ width: 9, height: 9, borderRadius: 999, background: meta.color }} />
          <span style={{ fontWeight: 700, fontSize: 13.5 }}>{meta.label}</span>
        </div>
        {isRuleBased ? (
          <Badge tone={data.status === 'Unavailable' ? 'neutral' : (statusLabel && isFlaggedStatus(data.status) ? 'warning' : 'success')}>
            {statusLabel || 'Signal'}
          </Badge>
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

      {Number.isFinite(Number(value)) && (
        <div className="meter">
          <div className="meter__fill" style={{ width: toPercent(value, 0), background: meta.color }} />
        </div>
      )}

      {(data?.explanation || meta.description) && <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{data?.explanation || meta.description}</div>}

      {meta.key === 'eye_blink' && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          {(supplementaryEvidence?.blink_events || []).length ? (
            <ul style={{ margin: 0, paddingLeft: 18 }}>{supplementaryEvidence.blink_events.map((event, index) => <li key={index}>Blink event {event.start_seconds == null ? '' : `at ${Number(event.start_seconds).toFixed(2)}s`}</li>)}</ul>
          ) : 'No timestamped blink events were returned.'}
        </div>
      )}
      {meta.key === 'lip_sync' && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          {(supplementaryEvidence?.lip_sync_window_evidence || []).length ? (
            <ul style={{ margin: 0, paddingLeft: 18 }}>{supplementaryEvidence.lip_sync_window_evidence.map((window, index) => <li key={index}>{Number(window.window_start_seconds).toFixed(1)}–{Number(window.window_end_seconds).toFixed(1)}s · mismatch {toPercent(window.mismatch_score, 0)}</li>)}</ul>
          ) : 'No timestamped lip-sync windows were returned.'}
        </div>
      )}

      {data?.status && (
        <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
          {data.status}
        </div>
      )}
    </div>
  )
}
