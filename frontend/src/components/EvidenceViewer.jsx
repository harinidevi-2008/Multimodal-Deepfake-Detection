import FrameViewer from './FrameViewer'
import EmptyState from './ui/EmptyState'

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
            <div style={{ fontWeight: 700, marginBottom: 8 }}>Selected frames</div>
            <FrameViewer frames={frames} />
          </div>
        ) : (
          <EmptyState
            title="Relevant frames unavailable"
            body="No localized visual evidence was returned for this result."
          />
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
            Audio and visual consistency signals were used in the overall verdict, but no localized audio or lip-sync proof was available for this result.
          </p>
        )}
      </div>
    </div>
  )
}
