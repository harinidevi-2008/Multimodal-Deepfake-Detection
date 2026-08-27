/**
 * Contract-shaped dev stand-in for the real backend — NOT the production
 * API. It lets the frontend's actual network/error-handling code path
 * (fetch, dev proxy, multipart upload, non-2xx error rendering) be
 * exercised end-to-end before the real Python backend
 * (inference/full_pipeline.py + an API adapter around it) exists.
 *
 * It returns the bundled mock JSON fixtures verbatim — it does not run any
 * model. Delete this once the real backend implements POST /api/analyze
 * per API_CONTRACT.md.
 *
 * Run: node scripts/dev-mock-server.mjs [port]
 * Then either:
 *   - `npm run dev` and use the app normally (vite proxies /api -> here), or
 *   - VITE_API_BASE_URL=http://localhost:8000 npm run dev
 *
 * Simulate specific backend error codes by adding a query string to the
 * upload... actually simplest: set MOCK_FAIL=<code> in the environment
 * before starting the server, e.g.:
 *   MOCK_FAIL=missing_checkpoint node scripts/dev-mock-server.mjs
 */
import { createServer } from 'node:http'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { dirname, join } from 'node:path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const PORT = Number(process.argv[2] || process.env.PORT || 8000)
const FORCED_FAILURE = process.env.MOCK_FAIL || null

const mockFake = JSON.parse(
  readFileSync(join(__dirname, '..', 'src', 'mock', 'mockResult.fake.json'), 'utf8')
)
const mockReal = JSON.parse(
  readFileSync(join(__dirname, '..', 'src', 'mock', 'mockResult.real.json'), 'utf8')
)

const FAILURE_RESPONSES = {
  missing_checkpoint: {
    status: 503,
    body: {
      error: 'missing_checkpoint',
      message: 'A required model checkpoint is missing on the server.',
      details: 'fusion/checkpoints/enhanced_fusion_best.pt not found (DFD_FUSION_CHECKPOINT).'
    }
  },
  unsupported_media_type: {
    status: 415,
    body: {
      error: 'unsupported_media_type',
      message: 'That file type is not supported — please upload a video.',
      details: null
    }
  },
  inference_failed: {
    status: 500,
    body: {
      error: 'inference_failed',
      message: 'Inference failed on the backend.',
      details: 'See server logs for the full traceback.'
    }
  }
}

let callCount = 0

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`)

  if (req.method === 'GET' && url.pathname === '/api/health') {
    return sendJson(res, 200, { status: 'ok' })
  }

  if (req.method === 'POST' && url.pathname === '/api/analyze') {
    const chunks = []
    for await (const chunk of req) chunks.push(chunk)
    const body = Buffer.concat(chunks)

    const contentType = req.headers['content-type'] || ''
    const looksMultipart = contentType.startsWith('multipart/form-data')
    const hasVideoField = looksMultipart && body.includes('name="video"')

    if (!looksMultipart || body.length === 0) {
      return sendJson(res, 400, {
        error: 'missing_video',
        message: 'No video file was provided.',
        details: null
      })
    }
    if (!hasVideoField) {
      return sendJson(res, 400, {
        error: 'missing_video',
        message: 'The "video" form field was not found in the upload.',
        details: null
      })
    }

    if (FORCED_FAILURE && FAILURE_RESPONSES[FORCED_FAILURE]) {
      const f = FAILURE_RESPONSES[FORCED_FAILURE]
      return sendJson(res, f.status, f.body)
    }

    // Simulate the pipeline taking a moment, then alternate fake/real so
    // both UI states are easy to eyeball across repeated uploads.
    await new Promise((r) => setTimeout(r, 1200))
    callCount += 1
    return sendJson(res, 200, callCount % 2 === 0 ? mockReal : mockFake)
  }

  sendJson(res, 404, { error: 'not_found', message: 'No such route.', details: null })
})

function sendJson(res, status, body) {
  const payload = JSON.stringify(body)
  res.writeHead(status, {
    'Content-Type': 'application/json',
    'Content-Length': Buffer.byteLength(payload),
    // Only needed if you point the frontend straight at this server without
    // the vite dev proxy (e.g. VITE_API_BASE_URL=http://localhost:8000 and
    // running vite on a different port). The proxy path needs none of this.
    'Access-Control-Allow-Origin': 'http://localhost:5173',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type'
  })
  res.end(payload)
}

server.listen(PORT, () => {
  console.log(`Dev mock backend (NOT the real pipeline) listening on http://localhost:${PORT}`)
  console.log('Implements: GET /api/health, POST /api/analyze — see API_CONTRACT.md')
  if (FORCED_FAILURE) console.log(`Forcing every /api/analyze call to fail as: ${FORCED_FAILURE}`)
})
