// Central place describing each modality: color, copy, and — critically —
// whether its number is a calibrated learned probability or a rule-based
// anomaly/mismatch score. Keeping this in one file means a label can't
// drift out of sync between the modality cards, charts, and tooltips.

export const MODALITY_META = {
  visual: {
    key: 'visual',
    label: 'Visual',
    color: 'var(--hue-visual)',
    kind: 'learned_probability',
    metricLabel: 'Fake probability',
    description: 'Frame-level classifier trained on visual artifacts and inconsistencies.'
  },
  audio: {
    key: 'audio',
    label: 'Audio',
    color: 'var(--hue-audio)',
    kind: 'learned_probability',
    metricLabel: 'Fake probability',
    description: 'Classifier trained on the audio track embedding.'
  },
  semantic: {
    key: 'semantic',
    label: 'Semantic',
    color: 'var(--hue-semantic)',
    kind: 'learned_probability',
    metricLabel: 'Fake probability',
    description: 'Classifier trained on transcript/semantic-content embeddings.'
  },
  eye_blink: {
    key: 'eye_blink',
    label: 'Eye Blink',
    color: 'var(--hue-blink)',
    kind: 'rule_based_score',
    metricLabel: 'Anomaly score',
    description: 'Rule-based analysis of blink rate, duration, and eye-aspect-ratio pattern.'
  },
  lip_sync: {
    key: 'lip_sync',
    label: 'Lip Sync',
    color: 'var(--hue-lipsync)',
    kind: 'rule_based_score',
    metricLabel: 'Mismatch score',
    description: 'Rule-based correlation between mouth motion and the audio envelope.'
  }
}

export const MODALITY_ORDER = ['visual', 'audio', 'semantic', 'eye_blink', 'lip_sync']

export const RULE_BASED_CAVEAT =
  "Eye-blink and lip-sync are rule-based anomaly/mismatch scores, not trained, calibrated probabilities — they're shown on the same 0–100 scale for comparison, but shouldn't be read as \"% chance of fake.\" They feed the fusion model as inputs, they aren't a display-time afterthought."
