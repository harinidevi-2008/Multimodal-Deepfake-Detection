export function asFiniteNumber(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

export function toPercent(value, digits = 0) {
  const number = asFiniteNumber(value)
  return number === null ? 'Unavailable' : `${(number * 100).toFixed(digits)}%`
}

export function formatSeconds(value) {
  const number = asFiniteNumber(value)
  return number === null ? 'Unavailable' : `${number.toFixed(1)}s`
}

export function formatTimestamp(value) {
  const number = asFiniteNumber(value)
  if (number === null) return 'Unavailable'
  const mins = Math.floor(number / 60)
  const secs = (number % 60).toFixed(1).padStart(4, '0')
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
  const number = asFiniteNumber(value)
  if (number === null) return 'accent'
  if (number >= 0.6) return 'danger'
  if (number >= 0.4) return 'warning'
  return 'success'
}
