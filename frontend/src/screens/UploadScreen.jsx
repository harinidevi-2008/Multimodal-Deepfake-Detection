import { useCallback, useRef, useState } from 'react'

// Must match exactly what the backend accepts (api/server.py's
// ALLOWED_EXTENSIONS, itself matching audio/src/config.py's
// SUPPORTED_EXTENSIONS - the audio stream, and therefore the whole
// 5-modal pipeline, is not verified beyond these three containers).
// Advertising a format here that the backend rejects would just move
// the same error from an immediate, friendly client-side message to a
// confusing round-trip through the real API.
const ACCEPTED_TYPES = ['video/mp4', 'video/x-msvideo', 'video/quicktime']
const ACCEPTED_EXTENSIONS_RE = /\.(mp4|avi|mov)$/i
const ACCEPTED_FORMATS_LABEL = 'MP4, AVI, or MOV'

export default function UploadScreen({ onAnalyze }) {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const acceptFile = useCallback((candidate) => {
    if (!candidate) return
    const looksLikeVideo =
      ACCEPTED_TYPES.includes(candidate.type) || ACCEPTED_EXTENSIONS_RE.test(candidate.name)
    if (!looksLikeVideo) {
      setError(`Please choose a video file (${ACCEPTED_FORMATS_LABEL}).`)
      return
    }
    setError(null)
    setFile(candidate)
    setPreviewUrl(URL.createObjectURL(candidate))
  }, [])

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      acceptFile(e.dataTransfer.files?.[0])
    },
    [acceptFile]
  )

  const handleBrowse = (e) => acceptFile(e.target.files?.[0])

  const clearFile = () => {
    setFile(null)
    setPreviewUrl(null)
    setError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="stack" style={{ gap: 'var(--space-6)', maxWidth: 720, margin: '0 auto' }}>
      <div className="stack" style={{ gap: 'var(--space-2)', textAlign: 'center' }}>
        <h1 style={{ fontSize: 28 }}>Check a video for signs of manipulation</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: 15 }}>
          Truthframe analyzes visual, audio, semantic, blink, and lip-sync signals together and
          shows you exactly what it found — not just a single score.
        </p>
      </div>

      <div
        className="card"
        style={{
          padding: 0,
          overflow: 'hidden',
          border: isDragging ? '2px dashed var(--accent)' : '1px solid var(--border)',
          background: isDragging ? 'var(--accent-soft)' : 'var(--surface)'
        }}
        onDragOver={(e) => {
          e.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
      >
        {!file ? (
          <div
            style={{
              padding: 'var(--space-8) var(--space-6)',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 'var(--space-4)',
              textAlign: 'center'
            }}
          >
            <div
              style={{
                width: 64,
                height: 64,
                borderRadius: 16,
                background: 'var(--accent-soft)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
            >
              <UploadGlyph />
            </div>
            <div>
              <p style={{ fontWeight: 700, fontSize: 16 }}>Drag & drop a video here</p>
              <p style={{ color: 'var(--text-muted)', fontSize: 13, marginTop: 4 }}>
                {ACCEPTED_FORMATS_LABEL} — up to a few minutes long
              </p>
            </div>
            <button className="btn btn-secondary" onClick={() => inputRef.current?.click()}>
              Browse files
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="video/mp4,video/x-msvideo,video/quicktime,.mp4,.avi,.mov"
              className="visually-hidden"
              onChange={handleBrowse}
            />
          </div>
        ) : (
          <div style={{ padding: 'var(--space-5)' }}>
            <div className="grid grid-cols-2" style={{ gap: 'var(--space-5)', alignItems: 'center' }}>
              <video
                src={previewUrl}
                controls
                style={{
                  width: '100%',
                  borderRadius: 'var(--radius-md)',
                  background: '#000',
                  maxHeight: 220
                }}
              />
              <div className="stack" style={{ gap: 'var(--space-3)' }}>
                <div>
                  <p style={{ fontWeight: 700, fontSize: 15, wordBreak: 'break-all' }}>{file.name}</p>
                  <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
                    {(file.size / (1024 * 1024)).toFixed(1)} MB
                  </p>
                </div>
                <div className="stack" style={{ gap: 'var(--space-2)' }}>
                  <button className="btn btn-primary" onClick={() => onAnalyze(file)}>
                    Analyze video
                  </button>
                  <button className="btn btn-ghost" onClick={clearFile}>
                    Choose a different file
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="callout" style={{ borderColor: 'var(--danger-soft-border)', background: 'var(--danger-soft)', color: 'var(--danger-strong)' }}>
          {error}
        </div>
      )}

      <div className="grid grid-cols-3" style={{ gap: 'var(--space-4)' }}>
        <FeatureNote title="5 independent signals" body="Visual, audio, and semantic classifiers plus blink and lip-sync analysis, fused together." />
        <FeatureNote title="Evidence, not a black box" body="Every verdict comes with available frames and analysis timelines that explain what it found." />
        <FeatureNote title="Honest about uncertainty" body="Rule-based anomaly scores are labeled as such — never dressed up as calibrated probabilities." />
      </div>
    </div>
  )
}

function FeatureNote({ title, body }) {
  return (
    <div className="card card--tight" style={{ boxShadow: 'none' }}>
      <p style={{ fontWeight: 700, fontSize: 13.5 }}>{title}</p>
      <p style={{ color: 'var(--text-secondary)', fontSize: 12.5, marginTop: 6, lineHeight: 1.55 }}>
        {body}
      </p>
    </div>
  )
}

function UploadGlyph() {
  return (
    <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
      <path
        d="M12 15V3m0 0L7 8m5-5l5 5"
        stroke="var(--accent)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M4 15v3a2 2 0 002 2h12a2 2 0 002-2v-3"
        stroke="var(--accent)"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}
