export function toPercent(value, digits = 0) {
  return `${(value * 100).toFixed(digits)}%`
}

export function formatSeconds(value) {
  return `${value.toFixed(1)}s`
}

export function formatTimestamp(value) {
  const mins = Math.floor(value / 60)
  const secs = (value % 60).toFixed(1).padStart(4, '0')
  return mins > 0 ? `${mins}:${secs}` : `${secs}s`
}

// Consistent color per modality key, used across cards + charts.
export const MODALITY_COLORS = {
  visual: 'var(--hue-visual)',
  audio: 'var(--hue-audio)',
  semantic: 'var(--hue-semantic)',
  eye_blink: 'var(--hue-blink)',
  lip_sync: 'var(--hue-lipsync)'
}

export function scoreTone(value) {
  if (value >= 0.6) return 'danger'
  if (value >= 0.4) return 'warning'
  return 'success'
}
