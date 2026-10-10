import FrameViewer from './FrameViewer'
import EmptyState from './ui/EmptyState'
import AudioTimeline from './AudioTimeline'

export default function EvidenceViewer({ evidence }) {
  const ev = evidence || {}
  const frames = Array.isArray(ev.frames) ? ev.frames.slice(0, 4) : []
  const hasFrames = frames.length > 0
  const audio = ev.audio_waveform

  return (
    <div className="card">
      <div className="stack" style={{ gap: 'var(--space-4)' }}>
        {hasFrames ? (
          <div>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Visual Evidence Frames</div>
            <FrameViewer frames={frames} />
          </div>
        ) : (
          <EmptyState
            title="No visual artifact was localized"
            body="No suspicious frame was identified. Any sampled frames are supporting context only."
          />
        )}

        {audio ? <AudioTimeline audioWaveform={audio} /> : (
          <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', margin: 0 }}>
            A global audio score is available, but no specific suspicious audio interval was localized.
          </p>
        )}

      </div>
    </div>
  )
}
