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
    metricLabel: 'Fake likelihood',
    description: 'Learned visual signal.'
  },
  audio: {
    key: 'audio',
    label: 'Audio',
    color: 'var(--hue-audio)',
    kind: 'learned_probability',
    metricLabel: 'Fake likelihood',
    description: 'Learned audio signal.'
  },
  semantic: {
    key: 'semantic',
    label: 'Semantic',
    color: 'var(--hue-semantic)',
    kind: 'learned_probability',
    metricLabel: 'Fake likelihood',
    description: 'Learned semantic signal.'
  },
  eye_blink: {
    key: 'eye_blink',
    label: 'Blink consistency',
    color: 'var(--hue-blink)',
    kind: 'rule_based_score',
    metricLabel: 'Consistency',
    description: 'Supporting blink cue.'
  },
  lip_sync: {
    key: 'lip_sync',
    label: 'Lip-sync consistency',
    color: 'var(--hue-lipsync)',
    kind: 'rule_based_score',
    metricLabel: 'Consistency',
    description: 'Supporting audio-visual cue.'
  }
}

export const MODALITY_ORDER = ['visual', 'audio', 'semantic', 'eye_blink', 'lip_sync']

export const RULE_BASED_CAVEAT =
  'Blink and lip-sync are supporting consistency cues, not learned fake probabilities.'
