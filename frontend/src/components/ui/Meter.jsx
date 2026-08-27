import { toPercent } from '../../utils/formatters'

const TONE_VARS = {
  danger: 'var(--danger)',
  warning: 'var(--warning)',
  success: 'var(--success)',
  accent: 'var(--accent)'
}

/**
 * A labeled horizontal meter bar: label — bar — numeric value.
 * `color` may be a token like 'danger' or a raw CSS color (e.g. a modality hue).
 */
export default function Meter({ label, value, color = 'accent', digits = 0 }) {
  const fillColor = TONE_VARS[color] || color
  return (
    <div className="meter-row">
      {label && <span className="meter-row__label">{label}</span>}
      <div className="meter" style={{ flex: 1 }}>
        <div
          className="meter__fill"
          style={{ width: toPercent(value, 0), background: fillColor }}
        />
      </div>
      <span className="meter-row__value">{toPercent(value, digits)}</span>
    </div>
  )
}
