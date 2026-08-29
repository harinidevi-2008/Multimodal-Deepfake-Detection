import { useState } from 'react'
import Badge from './ui/Badge'
import { formatSeconds } from '../utils/formatters'
import { resolveEvidenceUrl } from '../api/client'

const REASON_LABEL = {
  eye_blink_event: 'Blink event',
  uniform_sample: 'Supporting visual sample'
}

const EVIDENCE_TYPE_LABEL = {
  localized_visual_artifact: 'Localized visual artifact',
  supporting_context: 'Supporting visual sample',
  blink_event: 'Blink event',
  lip_sync_event: 'Lip-sync event',
  other: 'Other'
}

// Backend frame evidence may arrive as a data: URI (mock fixtures) or a
// server-relative evidence URL like "/api/evidence/<job_id>/<file>" (real
// backend, see API_CONTRACT.md) — either key resolves the same way.
function frameSrc(frame) {
  return resolveEvidenceUrl(frame.url || frame.image_url)
}

export default function FrameViewer({ frames }) {
  const [selected, setSelected] = useState(0)
  const frame = frames[selected]
  const src = frameSrc(frame)

  return (
    <div className="stack" style={{ gap: 'var(--space-4)' }}>
      <div
        style={{
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          border: '1px solid var(--border)',
          background: '#0d1220',
          position: 'relative'
        }}
      >
        {src ? (
          <img
            src={src}
            alt={`Frame at ${formatSeconds(frame.timestamp_seconds)}`}
            style={{ width: '100%', display: 'block', maxHeight: 340, objectFit: 'contain', margin: '0 auto' }}
          />
        ) : (
          <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 13 }}>
            Frame image unavailable
          </div>
        )}
        <div
          style={{
            position: 'absolute',
            top: 12,
            left: 12,
            display: 'flex',
            gap: 8
          }}
        >
          <Badge tone={frame.frame_selection_reason === 'eye_blink_event' ? 'warning' : 'neutral'}>
            {REASON_LABEL[frame.frame_selection_reason] || frame.frame_selection_reason}
          </Badge>
          <Badge tone={frame.evidence_type === 'localized_visual_artifact' ? 'warning' : 'neutral'}>
            {EVIDENCE_TYPE_LABEL[frame.evidence_type] || frame.evidence_type || 'Context'}
          </Badge>
          <Badge tone="neutral">frame #{frame.frame_index}</Badge>
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 11.5, color: 'var(--text-muted)' }}>
        <span><strong>Timestamp:</strong> {formatSeconds(frame.timestamp_seconds)}</span>
        <span><strong>Reason:</strong> {REASON_LABEL[frame.frame_selection_reason] || frame.frame_selection_reason || 'Supporting visual sample'}</span>
        <span><strong>Evidence:</strong> {EVIDENCE_TYPE_LABEL[frame.evidence_type] || frame.evidence_type || 'Supporting visual sample'}</span>
        {frame.confidence != null && <span><strong>Confidence:</strong> {Number(frame.confidence).toFixed(2)}</span>}
        {frame.strength && <span><strong>Strength:</strong> {frame.strength}</span>}
      </div>
      <p style={{ fontSize: 12, color: 'var(--text-muted)' }}>
        {frame.explanation || (frame.detected_artifact == null
          ? 'Visual artifact localization is unavailable. This is a supporting visual sample and is not proof of manipulation.'
          : frame.detected_artifact)}
      </p>

      <div style={{ display: 'flex', gap: 8, overflowX: 'auto', paddingBottom: 4 }}>
        {frames.map((f, i) => {
          const thumbSrc = frameSrc(f)
          return (
            <button
              key={f.frame_index}
              onClick={() => setSelected(i)}
              style={{
                flexShrink: 0,
                width: 84,
                height: 64,
                borderRadius: 'var(--radius-sm)',
                overflow: 'hidden',
                border: i === selected ? '2px solid var(--accent)' : '1px solid var(--border)',
                padding: 0,
                cursor: 'pointer',
                position: 'relative',
                background: 'var(--bg-subtle)'
              }}
            >
              {thumbSrc && (
                <img src={thumbSrc} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              )}
              {f.frame_selection_reason === 'eye_blink_event' && (
                <span
                  style={{
                    position: 'absolute',
                    bottom: 2,
                    right: 2,
                    width: 8,
                    height: 8,
                    borderRadius: 999,
                    background: 'var(--warning)'
                  }}
                />
              )}
            </button>
          )
        })}
      </div>
    </div>
  )
}
