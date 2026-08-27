import { useState } from 'react'

/**
 * A small "?" affordance that reveals a clarifying note on hover/focus.
 * Used throughout the dashboard to flag rule-based scores as
 * anomaly/mismatch scores rather than calibrated probabilities.
 */
export default function InfoTip({ text }) {
  const [open, setOpen] = useState(false)
  return (
    <span
      style={{ position: 'relative', display: 'inline-flex' }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      <span className="tooltip-icon" tabIndex={0} role="button" aria-label="More information">
        ?
      </span>
      {open && (
        <span
          role="tooltip"
          style={{
            position: 'absolute',
            bottom: 'calc(100% + 8px)',
            left: '50%',
            transform: 'translateX(-50%)',
            width: 240,
            background: 'var(--text-primary)',
            color: 'var(--text-inverse)',
            fontSize: 12,
            fontWeight: 500,
            lineHeight: 1.5,
            padding: '10px 12px',
            borderRadius: 'var(--radius-sm)',
            boxShadow: 'var(--shadow-lg)',
            zIndex: 20
          }}
        >
          {text}
        </span>
      )}
    </span>
  )
}
