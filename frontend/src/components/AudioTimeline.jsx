import { useState } from 'react'
import { formatSeconds } from '../utils/formatters'

const WIDTH = 900
const HEIGHT = 140

export default function AudioTimeline({ audioWaveform }) {
  const { envelope, duration_seconds } = audioWaveform
  const [hoverIndex, setHoverIndex] = useState(null)
  const barWidth = WIDTH / envelope.length
  const mid = HEIGHT / 2

  return (
    <div className="stack" style={{ gap: 'var(--space-2)' }}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        height={HEIGHT}
        style={{ display: 'block', overflow: 'visible' }}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <line x1={0} y1={mid} x2={WIDTH} y2={mid} stroke="var(--border)" strokeWidth={1} />
        {envelope.map((amp, i) => {
          const h = Math.max(2, amp * (HEIGHT / 2 - 8))
          const isHover = i === hoverIndex
          return (
            <rect
              key={i}
              x={i * barWidth}
              y={mid - h}
              width={Math.max(1, barWidth - 1)}
              height={h * 2}
              rx={1}
              fill={isHover ? 'var(--accent-strong)' : 'var(--hue-audio)'}
              opacity={isHover ? 1 : 0.75}
              onMouseEnter={() => setHoverIndex(i)}
            />
          )
        })}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }} className="mono">
        <span>0:00</span>
        {hoverIndex !== null && (
          <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
            {formatSeconds((hoverIndex / envelope.length) * duration_seconds)} · amplitude{' '}
            {envelope[hoverIndex].toFixed(2)}
          </span>
        )}
        <span>{formatSeconds(duration_seconds)}</span>
      </div>
    </div>
  )
}
