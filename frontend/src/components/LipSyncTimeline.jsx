import { useState } from 'react'
import { formatSeconds } from '../utils/formatters'

const WIDTH = 900
const HEIGHT = 150
const PAD_BOTTOM = 24

function colorFor(score) {
  if (score >= 0.6) return 'var(--danger)'
  if (score >= 0.4) return 'var(--warning)'
  return 'var(--hue-lipsync)'
}

export default function LipSyncTimeline({ windows }) {
  const [hoverIndex, setHoverIndex] = useState(null)
  const barWidth = WIDTH / windows.length
  const plotHeight = HEIGHT - PAD_BOTTOM
  const hovered = hoverIndex !== null ? windows[hoverIndex] : null

  return (
    <div className="stack" style={{ gap: 'var(--space-2)' }}>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        width="100%"
        height={HEIGHT}
        style={{ display: 'block', overflow: 'visible' }}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <line
          x1={0}
          y1={plotHeight - plotHeight * 0.5}
          x2={WIDTH}
          y2={plotHeight - plotHeight * 0.5}
          stroke="var(--border)"
          strokeDasharray="4 4"
        />
        {windows.map((w, i) => {
          const h = w.mismatch_score * plotHeight
          const isHover = i === hoverIndex
          return (
            <rect
              key={i}
              x={i * barWidth + 1}
              y={plotHeight - h}
              width={Math.max(1, barWidth - 2)}
              height={h}
              rx={2}
              fill={colorFor(w.mismatch_score)}
              opacity={isHover ? 1 : 0.85}
              onMouseEnter={() => setHoverIndex(i)}
            />
          )
        })}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)' }} className="mono">
        <span>{formatSeconds(windows[0].window_start_seconds)}</span>
        {hovered && (
          <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>
            {formatSeconds(hovered.window_start_seconds)}–{formatSeconds(hovered.window_end_seconds)} ·
            mismatch {hovered.mismatch_score.toFixed(2)}
          </span>
        )}
        <span>{formatSeconds(windows[windows.length - 1].window_end_seconds)}</span>
      </div>
    </div>
  )
}
