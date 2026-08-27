import { useState } from 'react'
import FrameViewer from './FrameViewer'
import AudioTimeline from './AudioTimeline'
import LipSyncTimeline from './LipSyncTimeline'
import BlinkTimeline from './BlinkTimeline'
import InfoTip from './ui/InfoTip'
import EmptyState from './ui/EmptyState'

const TABS = [
  { key: 'frames', label: 'Frames' },
  { key: 'audio', label: 'Audio waveform' },
  { key: 'lip_sync', label: 'Lip-sync mismatch' },
  { key: 'blink', label: 'Blink pattern' }
]

export default function EvidenceViewer({ evidence }) {
  const [tab, setTab] = useState('frames')
  const ev = evidence || {}

  const hasFrames = Array.isArray(ev.frames) && ev.frames.length > 0
  const hasAudio = ev.audio_waveform && Array.isArray(ev.audio_waveform.envelope) && ev.audio_waveform.envelope.length > 0
  const hasLipSyncWindows = Array.isArray(ev.lip_sync_window_evidence) && ev.lip_sync_window_evidence.length > 0
  const hasBlinkTimeline = Array.isArray(ev.blink_timeline) && ev.blink_timeline.length > 0

  return (
    <div className="card">
      <div className="card__header">
        <div>
          <div className="card__title">Evidence</div>
          <div className="card__subtitle">The frames and signal traces behind the verdict</div>
        </div>
        <div className="tabs">
          {TABS.map((t) => (
            <button
              key={t.key}
              className={`tab ${tab === t.key ? 'is-active' : ''}`}
              onClick={() => setTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {tab === 'frames' &&
        (hasFrames ? (
          <FrameViewer frames={ev.frames} />
        ) : (
          <EmptyState
            title="Frame evidence unavailable"
            body="No evidence frames were returned for this video."
          />
        ))}

      {tab === 'audio' &&
        (hasAudio ? (
          <div className="stack" style={{ gap: 'var(--space-3)' }}>
            <p style={{ fontSize: 12.5, color: 'var(--text-secondary)' }}>
              Amplitude envelope of the extracted audio track, used as input to the audio classifier.
            </p>
            <AudioTimeline audioWaveform={ev.audio_waveform} />
          </div>
        ) : (
          <EmptyState
            title="Audio waveform unavailable"
            body="The audio fake probability above still comes from the real classifier — a browser-ready waveform just wasn't generated for this request."
          />
        ))}

      {tab === 'lip_sync' &&
        (hasLipSyncWindows ? (
          <div className="stack" style={{ gap: 'var(--space-3)' }}>
            <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
              Per-window mismatch score between mouth motion and audio.
              <InfoTip text="Rule-based correlation score per ~0.8s window, not a trained probability. Higher bars mean weaker correlation between lip motion and audio in that window." />
            </p>
            <LipSyncTimeline windows={ev.lip_sync_window_evidence} />
          </div>
        ) : (
          <EmptyState
            title="Window evidence unavailable"
            body="The overall lip-sync mismatch score above was still computed — windowed evidence requires the raw video and could not be recovered for this request."
          />
        ))}

      {tab === 'blink' &&
        (hasBlinkTimeline ? (
          <div className="stack" style={{ gap: 'var(--space-3)' }}>
            <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 6 }}>
              Eye-aspect-ratio (EAR) over time — dips mark detected blink events.
              <InfoTip text="EAR is a geometric proxy from eye-landmark tracking, not a model probability. Markers show frames flagged as blink events by the rule-based detector." />
            </p>
            <BlinkTimeline blinkTimeline={ev.blink_timeline} />
          </div>
        ) : (
          <EmptyState
            title="Blink evidence unavailable"
            body="The blink anomaly score above was still computed — the per-frame timeline could not be extracted for this request."
          />
        ))}
    </div>
  )
}
