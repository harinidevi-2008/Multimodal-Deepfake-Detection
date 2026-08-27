/**
 * Generates the mock JSON fixtures used by the frontend before the real
 * inference API exists (see inference/full_pipeline.py + evidence/evidence_builder.py
 * in the model repo — this file's shape mirrors that schema so swapping in the
 * real API later is a drop-in, not a rewrite).
 *
 * Run: node scripts/generate-mock.mjs
 * Writes: src/mock/mockResult.fake.json, src/mock/mockResult.real.json
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const outDir = join(__dirname, '..', 'src', 'mock')
mkdirSync(outDir, { recursive: true })

// ---- deterministic PRNG so re-runs produce stable diffs ----
function mulberry32(seed) {
  return function () {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function placeholderFrameSVG({ index, label, tone }) {
  // Lightweight inline "face crop" placeholder — a gradient card with a
  // simple abstracted face glyph, so the UI has something to lay out and
  // crop/zoom against before real extracted frames exist.
  const tones = {
    neutral: ['#eef2f7', '#dbe3ee'],
    flag: ['#fde8e8', '#f8caca'],
    ok: ['#e7f6ee', '#c9ecd9']
  }
  const [c1, c2] = tones[tone] || tones.neutral
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240">
    <defs>
      <linearGradient id="g${index}" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="${c1}"/>
        <stop offset="1" stop-color="${c2}"/>
      </linearGradient>
    </defs>
    <rect width="320" height="240" rx="14" fill="url(#g${index})"/>
    <ellipse cx="160" cy="112" rx="54" ry="66" fill="#ffffff" opacity="0.55"/>
    <circle cx="136" cy="100" r="6" fill="#8894a8"/>
    <circle cx="184" cy="100" r="6" fill="#8894a8"/>
    <path d="M136 132 Q160 148 184 132" stroke="#8894a8" stroke-width="4" fill="none" stroke-linecap="round"/>
    <text x="160" y="220" font-family="Inter, sans-serif" font-size="13" font-weight="600" fill="#5b6472" text-anchor="middle">${label}</text>
  </svg>`
  return 'data:image/svg+xml;base64,' + Buffer.from(svg).toString('base64')
}

function buildScenario({ seed, verdict }) {
  const rand = mulberry32(seed)
  const isFake = verdict === 'LIKELY_DEEPFAKE'

  const durationSeconds = 14.8
  const fps = 30

  // ---- per-modality learned probabilities (classifier outputs) ----
  const base = isFake
    ? { visual: 0.81, audio: 0.64, semantic: 0.58 }
    : { visual: 0.12, audio: 0.19, semantic: 0.24 }
  const jitter = () => (rand() - 0.5) * 0.06
  const visual_fake_probability = clamp01(base.visual + jitter())
  const audio_fake_probability = clamp01(base.audio + jitter())
  const semantic_fake_probability = clamp01(base.semantic + jitter())

  // ---- rule-based signals: explicitly NOT probabilities (see caveat) ----
  const blink_anomaly_score = clamp01((isFake ? 0.74 : 0.18) + jitter())
  const lip_sync_mismatch_score = clamp01((isFake ? 0.69 : 0.15) + jitter())

  const final_fake_probability = clamp01(
    0.38 * visual_fake_probability +
      0.24 * audio_fake_probability +
      0.14 * semantic_fake_probability +
      0.13 * blink_anomaly_score +
      0.11 * lip_sync_mismatch_score
  )
  const final_real_probability = 1 - final_fake_probability
  const low_confidence = Math.abs(final_fake_probability - 0.5) < 0.08

  // ---- descriptive attention weights (non-causal) ----
  const rawAttn = {
    visual: 0.3 + rand() * 0.15,
    audio: 0.18 + rand() * 0.12,
    semantic: 0.12 + rand() * 0.1,
    eye_blink: 0.12 + rand() * 0.1,
    lip_sync: 0.14 + rand() * 0.1
  }
  const attnSum = Object.values(rawAttn).reduce((a, b) => a + b, 0)
  const attention_summary = {
    weights: Object.fromEntries(
      Object.entries(rawAttn).map(([k, v]) => [k, +(v / attnSum).toFixed(3)])
    ),
    note:
      "Descriptive cross-attention weights from the fusion model's attention layer. These describe where the model attended, not a causal or verified measure of importance — see modality_contributions for the ablation-based estimate."
  }

  // ---- ablation-based modality contributions (ΔP(fake) when zeroed) ----
  const contrib = {
    visual: isFake ? 0.24 + rand() * 0.05 : 0.05 + rand() * 0.03,
    audio: isFake ? 0.13 + rand() * 0.04 : 0.04 + rand() * 0.03,
    semantic: isFake ? 0.07 + rand() * 0.03 : 0.03 + rand() * 0.02,
    eye_blink: isFake ? 0.11 + rand() * 0.04 : 0.02 + rand() * 0.02,
    lip_sync: isFake ? 0.1 + rand() * 0.04 : 0.02 + rand() * 0.02
  }
  const modality_contributions = {
    deltas: Object.fromEntries(Object.entries(contrib).map(([k, v]) => [k, +v.toFixed(3)])),
    note:
      "Ablation-based: each value is the drop in the final fake probability when that modality's input is zeroed out. Zeroing a stream is not the same as the model having 'never observed' it — treat as a heuristic, not ground truth attribution."
  }

  // ---- evidence: frames ----
  const blinkEventFrames = isFake ? [42, 118, 267] : [55, 201]
  const totalFrames = Math.round(durationSeconds * fps)
  const uniformFrames = [0.12, 0.34, 0.58, 0.82, 0.95].map((f) =>
    Math.round(f * totalFrames)
  )
  const frameIndices = [...new Set([...blinkEventFrames, ...uniformFrames])].sort(
    (a, b) => a - b
  )
  const frames = frameIndices.map((idx, i) => {
    const isBlinkEvt = blinkEventFrames.includes(idx)
    const timestamp = +(idx / fps).toFixed(2)
    const tone = isBlinkEvt ? (isFake ? 'flag' : 'ok') : 'neutral'
    return {
      frame_index: idx,
      timestamp_seconds: timestamp,
      image_url: placeholderFrameSVG({
        index: i,
        label: `t=${timestamp}s`,
        tone
      }),
      frame_selection_reason: isBlinkEvt ? 'eye_blink_event' : 'uniform_sample',
      detected_artifact: null
    }
  })

  // ---- evidence: audio waveform (downsampled amplitude envelope) ----
  const waveformPoints = 260
  const audio_waveform = {
    sample_rate_hz: 16000,
    duration_seconds: durationSeconds,
    envelope: Array.from({ length: waveformPoints }, (_, i) => {
      const t = i / waveformPoints
      const speechEnvelope =
        0.35 +
        0.3 * Math.abs(Math.sin(t * 22 + rand() * 0.4)) +
        0.15 * Math.abs(Math.sin(t * 5))
      return +(speechEnvelope * (0.85 + rand() * 0.3)).toFixed(3)
    })
  }

  // ---- evidence: lip-sync window mismatch scores ----
  const windowSeconds = 0.8
  const windowCount = Math.floor(durationSeconds / windowSeconds)
  const mismatchHotspots = isFake ? [4, 5, 6, 12, 13] : []
  const lip_sync_window_evidence = Array.from({ length: windowCount }, (_, i) => {
    const hot = mismatchHotspots.includes(i)
    const scoreBase = hot ? 0.62 : isFake ? 0.32 : 0.14
    return {
      window_start_seconds: +(i * windowSeconds).toFixed(2),
      window_end_seconds: +((i + 1) * windowSeconds).toFixed(2),
      mismatch_score: clamp01(scoreBase + (rand() - 0.5) * 0.12)
    }
  })

  // ---- evidence: blink timeline (EAR = eye-aspect-ratio proxy) ----
  const blinkSamples = Math.round(durationSeconds * 10) // 10Hz
  let ear = 0.32
  const blink_timeline = []
  const blinkEventTimes = blinkEventFrames.map((f) => f / fps)
  for (let i = 0; i < blinkSamples; i++) {
    const t = +(i / 10).toFixed(2)
    const nearBlink = blinkEventTimes.some((bt) => Math.abs(bt - t) < 0.15)
    ear = nearBlink ? 0.09 + rand() * 0.03 : 0.3 + (rand() - 0.5) * 0.05
    blink_timeline.push({
      timestamp_seconds: t,
      ear_value: +ear.toFixed(3),
      is_blink: nearBlink
    })
  }

  const contextual_analysis = isFake
    ? {
        detected_mismatch:
          'Lip movements do not consistently track the audio track between roughly 3.2s and 5.6s, and again near 9.6s–10.4s.',
        explanation:
          'The lip-sync analyzer found sustained low correlation between mouth-shape motion and the audio envelope in these windows, coinciding with two of the three irregular blink events. Individually, none of these signals is conclusive — together, the fusion model weighted them as the strongest evidence toward the deepfake verdict.'
      }
    : {
        detected_mismatch: 'No sustained lip-sync or blink-pattern anomalies were detected.',
        explanation:
          'Lip movement stayed correlated with the audio track throughout, and blink timing and duration fell within the expected range. Visual, audio, and semantic streams were all classified with low fake probability.'
      }

  return {
    video_filename: isFake ? 'clip_0417_interview.mp4' : 'clip_0203_statement.mp4',
    video_duration_seconds: durationSeconds,
    analyzed_at: '2026-08-27T10:42:00+05:30',
    processing_time_seconds: isFake ? 9.4 : 7.1,
    final_verdict: verdict,
    final_fake_probability: +final_fake_probability.toFixed(3),
    final_real_probability: +final_real_probability.toFixed(3),
    low_confidence,
    model_confidence_note:
      'Model confidence reflects distance from the 0.5 decision boundary on the calibrated fusion output, not a calibrated probability of ground truth.',
    modalities: {
      visual: {
        label: 'Visual',
        kind: 'learned_probability',
        fake_probability: +visual_fake_probability.toFixed(3)
      },
      audio: {
        label: 'Audio',
        kind: 'learned_probability',
        fake_probability: +audio_fake_probability.toFixed(3)
      },
      semantic: {
        label: 'Semantic',
        kind: 'learned_probability',
        fake_probability: +semantic_fake_probability.toFixed(3)
      },
      eye_blink: {
        label: 'Eye Blink',
        kind: 'rule_based_score',
        anomaly_score: +blink_anomaly_score.toFixed(3),
        status: blink_anomaly_score > 0.5 ? 'Irregular' : 'Normal'
      },
      lip_sync: {
        label: 'Lip Sync',
        kind: 'rule_based_score',
        mismatch_score: +lip_sync_mismatch_score.toFixed(3),
        status: lip_sync_mismatch_score > 0.5 ? 'Inconsistent' : 'Consistent'
      }
    },
    attention_summary,
    modality_contributions,
    evidence: {
      frames,
      audio_waveform,
      lip_sync_window_evidence,
      blink_timeline
    },
    contextual_analysis
  }
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v))
}

const fake = buildScenario({ seed: 42, verdict: 'LIKELY_DEEPFAKE' })
const real = buildScenario({ seed: 7, verdict: 'LIKELY_REAL' })

writeFileSync(join(outDir, 'mockResult.fake.json'), JSON.stringify(fake, null, 2))
writeFileSync(join(outDir, 'mockResult.real.json'), JSON.stringify(real, null, 2))

console.log('Wrote mockResult.fake.json and mockResult.real.json to src/mock/')
