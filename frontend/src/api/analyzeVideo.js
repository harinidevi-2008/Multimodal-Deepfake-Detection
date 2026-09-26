import { apiUrl, AnalysisError, USE_MOCK_API } from './client'
import mockFake from '../mock/mockResult.fake.json'

// These stages are emitted by the backend job. Their ordering only renders
// active/completed states; the backend remains the source of truth.
export const ANALYSIS_STAGES = [
  { key: 'queued', label: 'Queued for analysis' },
  { key: 'probing_video', label: 'Reading video metadata' },
  { key: 'extracting_visual', label: 'Analyzing visual stream...' },
  { key: 'extracting_audio', label: 'Analyzing audio...' },
  { key: 'extracting_semantic', label: 'Extracting semantic features...' },
  { key: 'extracting_blink', label: 'Analyzing eye-blink patterns...' },
  { key: 'extracting_lipsync', label: 'Checking lip synchronization...' },
  { key: 'running_classifiers', label: 'Running learned classifiers...' },
  { key: 'running_fusion', label: 'Fusing modalities and scoring...' },
  { key: 'serializing', label: 'Building evidence...' }
]

const STATUS_POLL_INTERVAL_MS = 800

export async function analyzeVideo(file, onStatus) {
  if (USE_MOCK_API) {
    onStatus?.({ status: 'completed', stage: 'completed', progress: 100, message: 'Analysis complete.' })
    return mockFake
  }

  if (!file) {
    throw new AnalysisError('No video file was provided.', { code: 'missing_video' })
  }

  const formData = new FormData()
  formData.append('video', file)

  let response
  try {
    response = await fetch(apiUrl('/api/analyze'), { method: 'POST', body: formData })
  } catch (networkErr) {
    throw new AnalysisError("Couldn't reach the analysis backend. Is the API server running?", {
      code: 'network_error', details: networkErr?.message
    })
  }
  if (!response.ok) throw await toAnalysisError(response)

  let job
  try {
    job = await response.json()
  } catch (parseErr) {
    throw new AnalysisError('The backend returned a response that could not be parsed.', {
      code: 'invalid_response', status: response.status, details: parseErr?.message
    })
  }
  if (!job?.job_id) {
    throw new AnalysisError('The backend did not return an analysis job.', { code: 'invalid_response' })
  }

  return pollJobStatus(job.job_id, onStatus)
}

async function pollJobStatus(jobId, onStatus) {
  while (true) {
    let response
    try {
      response = await fetch(apiUrl(`/api/analyze/${jobId}/status`))
    } catch (networkErr) {
      throw new AnalysisError("Couldn't reach the analysis backend. Is the API server running?", {
        code: 'network_error', details: networkErr?.message
      })
    }
    if (!response.ok) throw await toAnalysisError(response)

    const status = await response.json()
    onStatus?.(status)
    if (status.status === 'completed') return status.result
    if (status.status === 'failed') {
      throw new AnalysisError(status.error?.message || 'Analysis failed on the backend.', {
        code: status.error?.error || 'backend_error', details: status.error?.details
      })
    }
    await new Promise((resolve) => setTimeout(resolve, STATUS_POLL_INTERVAL_MS))
  }
}

async function toAnalysisError(response) {
  let body = null
  try {
    body = await response.json()
  } catch {
    // Backend did not return JSON; fall through to the generic message.
  }
  if (body?.error) {
    return new AnalysisError(body.message || describeErrorCode(body.error), {
      code: body.error, status: response.status, details: body.details
    })
  }
  return new AnalysisError(`The backend returned an error (HTTP ${response.status}).`, {
    code: 'backend_error', status: response.status
  })
}

function describeErrorCode(code) {
  const known = {
    missing_checkpoint: 'A required model checkpoint is missing on the server.',
    unsupported_media_type: 'That file type is not supported - please upload a video.',
    corrupt_video: 'The video could not be read - it may be corrupt or an unsupported codec.',
    feature_extraction_failed: 'Feature extraction failed while processing the video.',
    inference_failed: 'Inference failed on the backend.'
  }
  return known[code] || 'The backend reported an error.'
}
