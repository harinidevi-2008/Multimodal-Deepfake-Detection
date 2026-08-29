import { useState } from 'react'
import { formatSeconds } from '../utils/formatters'
import { resolveEvidenceUrl } from '../api/client'

const WIDTH = 900
const HEIGHT = 140

export default function AudioTimeline({ audioWaveform }) {
  const { envelope, duration_seconds, localized_windows = [] } = audioWaveform
  const [hoverIndex, setHoverIndex] = useState(null)
  const [selectedWindow, setSelectedWindow] = useState(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [paused, setPaused] = useState(true)
  const barWidth = WIDTH / envelope.length
  const mid = HEIGHT / 2
  const mediaUrl = resolveEvidenceUrl(audioWaveform.media_url)
  const hasLocalizedWindows = Array.isArray(localized_windows) && localized_windows.length > 0

  function playWindow(window, audio) {
    if (!audio || !window) return
    setSelectedWindow(window)
    audio.currentTime = window.start_seconds
    audio.play()
  }

  return (
    <div className="stack" style={{ gap: 'var(--space-2)' }}>
      {mediaUrl && (
        <audio
          id="evidence-audio-player"
          src={mediaUrl}
          controls
          preload="metadata"
          style={{ width: '100%' }}
          onTimeUpdate={(event) => {
            const audio = event.currentTarget
            setCurrentTime(audio.currentTime)
            if (selectedWindow && audio.currentTime >= selectedWindow.end_seconds) {
              audio.pause()
            }
          }}
          onPlay={() => setPaused(false)}
          onPause={() => setPaused(true)}
        />
      )}
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
          const t = (i / envelope.length) * duration_seconds
          const inSelected = selectedWindow && t >= selectedWindow.start_seconds && t <= selectedWindow.end_seconds
          return (
            <rect
              key={i}
              x={i * barWidth}
              y={mid - h}
              width={Math.max(1, barWidth - 1)}
              height={h * 2}
              rx={1}
              fill={inSelected ? 'var(--warning)' : (isHover ? 'var(--accent-strong)' : 'var(--hue-audio)')}
              opacity={isHover || inSelected ? 1 : 0.75}
              onMouseEnter={() => setHoverIndex(i)}
            />
          )
        })}
        {hasLocalizedWindows && localized_windows.map((window) => {
          const x = (window.start_seconds / duration_seconds) * WIDTH
          const w = ((window.end_seconds - window.start_seconds) / duration_seconds) * WIDTH
          return (
            <rect
              key={`${window.start_seconds}-${window.end_seconds}`}
              x={x}
              y={0}
              width={Math.max(2, w)}
              height={HEIGHT}
              fill="var(--warning)"
              opacity={0.18}
              onClick={() => playWindow(window, document.getElementById('evidence-audio-player'))}
              style={{ cursor: mediaUrl ? 'pointer' : 'default' }}
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
        <span>{formatSeconds(currentTime)} / {formatSeconds(duration_seconds)}</span>
      </div>
      {hasLocalizedWindows ? (
        <div className="stack" style={{ gap: 'var(--space-2)' }}>
          {localized_windows.map((window) => (
            <button
              key={`${window.start_seconds}-${window.end_seconds}`}
              className="btn btn-secondary btn-sm"
              onClick={() => playWindow(window, document.getElementById('evidence-audio-player'))}
            >
              {paused ? 'Play' : 'Replay'} {formatSeconds(window.start_seconds)}-{formatSeconds(window.end_seconds)}
            </button>
          ))}
        </div>
      ) : (
        <p style={{ fontSize: 12, color: 'var(--text-muted)', margin: 0 }}>
          Audio evidence is global only. No timestamped suspicious audio interval was produced by the backend.
        </p>
      )}
    </div>
  )
}
