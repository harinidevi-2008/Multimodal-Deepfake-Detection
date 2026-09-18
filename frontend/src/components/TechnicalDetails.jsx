import InfoTip from './ui/InfoTip'
import Meter from './ui/Meter'
import { MODALITY_META, MODALITY_ORDER } from './modalityMeta'
import { formatSeconds, toPercent } from '../utils/formatters'

export default function TechnicalDetails({ result }) {
  const { attention_summary, modality_contributions, processing_time_seconds } = result
  const diagnostics = result.fusion_diagnostics
  const attentionWeights = attention_summary?.weights || {}
  const contributionDeltas = modality_contributions?.deltas || {}
  const decisionThreshold = result.decision_threshold ?? diagnostics?.decision_threshold
  const thresholdSource = diagnostics?.decision_threshold_source

  return (
    <div className="card">
      <div className="card__header">
        <div>
          <div className="card__title">Advanced diagnostics</div>
          <div className="card__subtitle">Calibrated threshold, timing, and per-modality contribution</div>
        </div>
      </div>

      <div className="grid grid-cols-2" style={{ gap: 'var(--space-6)' }}>
        <div className="stack" style={{ gap: 'var(--space-4)' }}>
          <div>
            <div className="kv-row">
              <span className="kv-row__key">Processing time</span>
              <span className="kv-row__value">{formatSeconds(processing_time_seconds)}</span>
            </div>
            <div className="kv-row">
              <span className="kv-row__key">Final fake probability</span>
              <span className="kv-row__value">{toPercent(result.final_fake_probability, 1)}</span>
            </div>
            <div className="kv-row">
              <span className="kv-row__key">Calibrated threshold</span>
              <span className="kv-row__value">{toPercent(decisionThreshold, 1)}</span>
            </div>
            {thresholdSource && (
              <div className="kv-row">
                <span className="kv-row__key">Threshold source</span>
                <span className="kv-row__value">
                  {thresholdSource.status || 'unknown'}
                  {thresholdSource.key ? ` (${thresholdSource.key})` : ''}
                </span>
              </div>
            )}
            <div className="kv-row">
              <span className="kv-row__key">Low-confidence flag</span>
              <span className="kv-row__value">{result.low_confidence ? 'Yes' : 'No'}</span>
            </div>
            {diagnostics?.class_indices && (
              <div className="kv-row">
                <span className="kv-row__key">Class indices</span>
                <span className="kv-row__value">real={diagnostics.class_indices.real}, fake={diagnostics.class_indices.fake}</span>
              </div>
            )}
            {diagnostics?.raw_logits && (
              <div className="kv-row">
                <span className="kv-row__key">Raw logits</span>
                <span className="kv-row__value">[{diagnostics.raw_logits.map((v) => Number.isFinite(Number(v)) ? Number(v).toFixed(3) : 'n/a').join(', ')}]</span>
              </div>
            )}
          </div>

          <div>
            <p style={{ fontSize: 13, fontWeight: 700, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
              Model diagnostics - attention weights
              <InfoTip text={attention_summary?.note || 'Attention diagnostics were not returned by the backend.'} />
            </p>
            <div className="stack" style={{ gap: 'var(--space-2)' }}>
              {MODALITY_ORDER.map((key) => (
                <Meter
                  key={key}
                  label={MODALITY_META[key].label}
                  value={attentionWeights[key]}
                  color={MODALITY_META[key].color}
                  digits={1}
                />
              ))}
            </div>
          </div>
        </div>

        <div>
          <p style={{ fontSize: 13, fontWeight: 700, marginBottom: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            Modality contribution (ablation)
            <InfoTip text={modality_contributions?.note || 'Ablation contribution diagnostics were not returned by the backend.'} />
          </p>
          <div className="stack" style={{ gap: 'var(--space-2)' }}>
            {MODALITY_ORDER.map((key) => (
              <Meter
                key={key}
                label={MODALITY_META[key].label}
                value={contributionDeltas[key]}
                color={MODALITY_META[key].color}
                digits={1}
              />
            ))}
          </div>
          <p style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 'var(--space-3)', lineHeight: 1.5 }}>
            Each bar is the drop in final fake probability when that modality is zeroed out —
            larger bars mean that stream mattered more to this specific verdict.
          </p>
        </div>
      </div>
    </div>
  )
}
