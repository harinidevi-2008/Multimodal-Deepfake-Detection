import FrameViewer from './FrameViewer'
import EmptyState from './ui/EmptyState'
import AudioTimeline from './AudioTimeline'

export default function EvidenceViewer({ evidence }) {
  const ev = evidence || {}
  const frames = Array.isArray(ev.frames) ? ev.frames.slice(0, 4) : []
  const hasFrames = frames.length > 0
  const hasLocalizedLipSync = Array.isArray(ev.lip_sync_window_evidence) && ev.lip_sync_window_evidence.length > 0

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
            title="Visual evidence frames unavailable"
            body="No selected context frames were returned for this result."
          />
        )}

        {ev.audio_waveform ? (
          <div>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Audio evidence</div>
            <AudioTimeline audioWaveform={ev.audio_waveform} />
          </div>
        ) : ev.audio_evidence_status === 'unavailable' ? (
          <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', margin: 0 }}>
            Audio analysis: Not available. Reason: {ev.audio_unavailable_reason || 'No audio stream detected'}.
          </p>
        ) : null}

        {ev.semantic_context?.transcript && (
          <div>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Transcript</div>
            <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', margin: 0 }}>
              {ev.semantic_context.transcript}
            </p>
          </div>
        )}

        {hasLocalizedLipSync ? (
          <div>
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Lip-sync inconsistency</div>
            <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', margin: 0 }}>
              Specific lip-sync windows flagged by the rule-based analyzer.
            </p>
          </div>
        ) : (
          <p style={{ fontSize: 12.5, color: 'var(--text-secondary)', margin: 0 }}>
            Audio/visual evidence is global unless timestamped evidence is available; no localized lip-sync proof was returned.
          </p>
        )}
      </div>
    </div>
  )
}
