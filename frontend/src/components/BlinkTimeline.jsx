import { useState } from 'react'
import { formatSeconds } from '../utils/formatters'

const WIDTH = 900
const HEIGHT = 150
const PAD = 20

export default function BlinkTimeline({ blinkTimeline }) {
  const [hoverIndex, setHoverIndex] = useState(null)
  const maxEar = Math.max(...blinkTimeline.map((p) => p.ear_value), 0.4)
  const duration = blinkTimeline[blinkTimeline.length - 1].timestamp_seconds

  const x = (i) => (i / (blinkTimeline.length - 1)) * (WIDTH - PAD * 2) + PAD
  const y = (v) => HEIGHT - PAD - (v / maxEar) * (HEIGHT - PAD * 2)

  const linePath = blinkTimeline
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${x(i).toFixed(1)} ${y(p.ear_value).toFixed(1)}`)
    .join(' ')

  const blinkEvents = blinkTimeline.filter((p) => p.is_blink)
  const hovered = hoverIndex !== null ? blinkTimeline[hoverIndex] : null

  return (
    <div className="stack" style={{ gap: 'var(--space-2)' }}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        height={HEIGHT}
        style={{ display: 'block', overflow: 'visible' }}
        onMouseLeave={() => setHoverIndex(null)}
      >
        {blinkEvents.map((p, i) => (
          <rect
            key={i}
            x={x(blinkTimeline.indexOf(p)) - 3}
            y={PAD}
            width={6}
            height={HEIGHT - PAD * 2}
            fill="var(--hue-blink)"
            opacity={0.14}
          />
        ))}

        <line x1={PAD} y1={HEIGHT - PAD} x2={WIDTH - PAD} y2={HEIGHT - PAD} stroke="var(--border)" />

        <path d={linePath} fill="none" stroke="var(--hue-blink)" strokeWidth={2} />

        {blinkTimeline.map((p, i) =>
          p.is_blink ? (
            <circle key={i} cx={x(i)} cy={y(p.ear_value)} r={3.5} fill="var(--warning-strong)" />
          ) : null
        )}

        {/* invisible hover targets */}
        {blinkTimeline.map((p, i) => (
          <rect
            key={`hover-${i}`}
            x={x(i) - (WIDTH / blinkTimeline.length) / 2}
            y={0}
            width={WIDTH / blinkTimeline.length}
            height={HEIGHT}
            fill="transparent"
            onMouseEnter={() => setHoverIndex(i)}
          />
        ))}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }} className="mono">
        <span>0:00</span>
        {hovered && (
          <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
            {formatSeconds(hovered.timestamp_seconds)} · EAR {hovered.ear_value.toFixed(2)}
            {hovered.is_blink ? ' · blink' : ''}
          </span>
        )}
        <span>{formatSeconds(duration)}</span>
      </div>
    </div>
  )
}
