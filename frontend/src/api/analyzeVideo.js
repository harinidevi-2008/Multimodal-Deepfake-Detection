// Real analysis API call.
//
// POSTs the uploaded video to POST /api/analyze (multipart/form-data,
// field name "video") and resolves with the backend's JSON response.
// See ../../API_CONTRACT.md for the exact response shape this — and every
// component downstream — expects, and for what the backend adapter is
// responsible for producing from the real pipeline result.
//
// No filename-based or random result selection lives in this file anymore.
// The only non-network path is the explicit VITE_USE_MOCK_API=true escape
// hatch for frontend-only development (see client.js) — off by default.

import { apiUrl, AnalysisError, USE_MOCK_API } from './client'
import mockFake from '../mock/mockResult.fake.json'

// Stages surfaced on the Analysis screen. The current backend integration
// runs the whole pipeline as a single synchronous request (see
// AnalysisScreen.jsx's note to the user), so these are an *indicative*
// progression, not per-stage server callbacks — this file advances them on
// a timer while the request is in flight, and always resolves the last one
// only once the real response has actually arrived.
export const ANALYSIS_STAGES = [
  { key: 'visual', label: 'Analyzing visual stream…' },
  { key: 'audio', label: 'Analyzing audio…' },
  { key: 'lip_sync', label: 'Checking lip synchronization…' },
  { key: 'eye_blink', label: 'Analyzing eye-blink patterns…' },
  { key: 'fusion', label: 'Fusing modalities and scoring…' }
]

const STAGE_INTERVAL_MS = 1400

/**
 * @param {File} file - the uploaded video file.
 * @param {(stageKey: string) => void} onStage - called as each indicative
 *   stage is reached, so the Analysis screen can advance its checklist.
 * @returns {Promise<object>} resolves to the backend's analysis result
 *   (see API_CONTRACT.md). Rejects with an AnalysisError on any failure —
 *   never resolves with fabricated/default data.
 */
export async function analyzeVideo(file, onStage) {
  if (USE_MOCK_API) {
    return mockAnalyze(onStage)
  }

  if (!file) {
    throw new AnalysisError('No video file was provided.', { code: 'missing_video' })
  }

  const stageTimer = driveIndicativeStages(onStage)

  try {
    const formData = new FormData()
    formData.append('video', file)

    let response
    try {
      response = await fetch(apiUrl('/api/analyze'), { method: 'POST', body: formData })
    } catch (networkErr) {
      throw new AnalysisError(
        "Couldn't reach the analysis backend. Is the API server running?",
        { code: 'network_error', details: networkErr?.message }
      )
    }

    if (!response.ok) {
      throw await toAnalysisError(response)
    }

    let data
    try {
      data = await response.json()
    } catch (parseErr) {
      throw new AnalysisError('The backend returned a response that could not be parsed.', {
        code: 'invalid_response',
        status: response.status,
        details: parseErr?.message
      })
    }

    return data
  } finally {
    stageTimer.stop()
    onStage?.('fusion')
  }
}

// Advances onStage(key) through the first four stages on a fixed cadence
// while the single backend request is in flight. This is explicitly an
// estimate — see AnalysisScreen.jsx copy — not a claim that the backend
// actually finished that stage. Stopped as soon as the real response (or
// error) comes back.
function driveIndicativeStages(onStage) {
  let i = 0
  onStage?.(ANALYSIS_STAGES[0].key)
  const id = setInterval(() => {
    i += 1
    if (i < ANALYSIS_STAGES.length - 1) {
      onStage?.(ANALYSIS_STAGES[i].key)
    } else {
      clearInterval(id)
    }
  }, STAGE_INTERVAL_MS)
  return { stop: () => clearInterval(id) }
}

async function toAnalysisError(response) {
  let body = null
  try {
    body = await response.json()
  } catch {
    // backend didn't return JSON — fall through to a generic message below
  }
  if (body?.error) {
    return new AnalysisError(body.message || describeErrorCode(body.error), {
      code: body.error,
      status: response.status,
      details: body.details
    })
  }
  return new AnalysisError(`The backend returned an error (HTTP ${response.status}).`, {
    code: 'backend_error',
    status: response.status
  })
}

function describeErrorCode(code) {
  const known = {
    missing_checkpoint: 'A required model checkpoint is missing on the server.',
    unsupported_media_type: 'That file type is not supported — please upload a video.',
    corrupt_video: 'The video could not be read — it may be corrupt or an unsupported codec.',
    feature_extraction_failed: 'Feature extraction failed while processing the video.',
    inference_failed: 'Inference failed on the backend.'
  }
  return known[code] || 'The backend reported an error.'
}

// ---- VITE_USE_MOCK_API=true path only (see client.js) ----
function mockAnalyze(onStage) {
  return new Promise((resolve) => {
    let i = 0
    const tick = () => {
      if (i < ANALYSIS_STAGES.length) {
        onStage?.(ANALYSIS_STAGES[i].key)
        i += 1
        setTimeout(tick, 500)
      } else {
        setTimeout(() => resolve(mockFake), 250)
      }
    }
    tick()
  })
}
