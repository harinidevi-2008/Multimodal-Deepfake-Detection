import Badge from './ui/Badge'
import InfoTip from './ui/InfoTip'
import { formatSeconds, toPercent } from '../utils/formatters'

export default function OverallResultPanel({ result }) {
  const isFake = result.final_verdict === 'LIKELY_DEEPFAKE'
  const fakePct = result.final_fake_probability
  const realPct = result.final_real_probability
  const decisionThreshold = result.decision_threshold ?? result.fusion_diagnostics?.decision_threshold
  const decisionThresholdLabel = toPercent(decisionThreshold, 1)
  const decisionThresholdTip = decisionThresholdLabel === 'Unavailable'
    ? 'The calibrated decision boundary was not returned for this result; confidence is still not a guarantee of ground truth.'
    : `Distance from the calibrated ${decisionThresholdLabel} decision boundary indicates how decisively the fusion model landed on this side of the line; it is not a guarantee of ground truth.`
  const showReview = result.review_recommended || result.evidence_consistency === 'LOW'

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
      <div
        className="grid grid-cols-2"
        style={{ gap: 'var(--space-6)', alignItems: 'center' }}
      >
        <div className="stack" style={{ gap: 'var(--space-3)' }}>
          <span style={{ fontSize: 12.5, fontWeight: 700, letterSpacing: '0.04em', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
            Overall result
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-3)', flexWrap: 'wrap' }}>
            <h2 style={{ fontSize: 30, color: isFake ? 'var(--danger-strong)' : 'var(--success-strong)' }}>
              {isFake ? 'Likely Deepfake' : 'Likely Real'}
            </h2>
            {result.low_confidence && (
              <Badge tone="warning" dot>
                Low confidence
              </Badge>
            )}
            {showReview && (
              <Badge tone="warning" dot>
                Review recommended
              </Badge>
            )}
          </div>
          <p style={{ color: 'var(--text-secondary)', fontSize: 13.5, maxWidth: 420 }}>
            {result.video_filename || 'Uploaded video'} - {formatSeconds(result.video_duration_seconds)} analyzed in{' '}
            {formatSeconds(result.processing_time_seconds)}
          </p>
          {(result.evidence_consistency || result.modality_disagreement) && (
            <div className="callout" style={{ fontSize: 12.5 }}>
              <span className="callout-strong">Evidence consistency: </span>
              {result.evidence_consistency || 'unknown'}
              <span style={{ marginLeft: 12 }} />
              <span className="callout-strong">Modality disagreement: </span>
              {result.modality_disagreement || 'unknown'}
            </div>
          )}
        </div>

        <div className="stack" style={{ gap: 'var(--space-3)' }}>
          <ProbabilityBar label="Fake probability" value={fakePct} tone="danger" />
          <ProbabilityBar label="Real probability" value={realPct} tone="success" />
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
              {result.model_confidence_note}
            </span>
            <InfoTip text={decisionThresholdTip} />
          </div>
        </div>
      </div>
    </div>
  )
}

function ProbabilityBar({ label, value, tone }) {
  const colorVar = tone === 'danger' ? 'var(--danger)' : 'var(--success)'
  const width = toPercent(value, 0)
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 6 }}>
        <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{label}</span>
        <span className="mono" style={{ fontWeight: 700, color: colorVar }}>
          {toPercent(value, 1)}
        </span>
      </div>
      <div className="meter" style={{ height: 10 }}>
        <div className="meter__fill" style={{ width: width === 'Unavailable' ? '0%' : width, background: colorVar }} />
      </div>
    </div>
  )
}
