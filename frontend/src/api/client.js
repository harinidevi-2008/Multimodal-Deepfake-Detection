// Small shared helpers for talking to the backend API.
//
// VITE_API_BASE_URL: leave unset for local dev — requests go to relative
// `/api/...` paths, which vite.config.js proxies to the backend
// (default http://localhost:8000). Set it explicitly for a production
// build where the frontend and backend are on different origins.
//
// VITE_USE_MOCK_API: "true" makes analyzeVideo() resolve from the bundled
// mock JSON instead of calling the network — useful for UI-only work with
// no backend running. Defaults to false (real backend) everywhere else.

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/+$/, '')
export const USE_MOCK_API = import.meta.env.VITE_USE_MOCK_API === 'true'

export function apiUrl(path) {
  const p = path.startsWith('/') ? path : `/${path}`
  return `${API_BASE_URL}${p}`
}

// Evidence fields from the backend are expected to be either an absolute
// URL, a data: URI (mock fixtures use these), or a server-relative path
// like "/api/evidence/<job_id>/<file>". Anything already absolute/data is
// passed through untouched; a relative path is resolved against the
// configured API base so it works whether or not the dev proxy is in play.
export function resolveEvidenceUrl(pathOrUrl) {
  if (!pathOrUrl) return null
  if (/^(data:|https?:\/\/|blob:)/.test(pathOrUrl)) return pathOrUrl
  return apiUrl(pathOrUrl)
}

// Structured error so the UI can distinguish "backend told us something
// specific" (missing checkpoint, bad upload, inference failure) from a
// generic network/parsing failure, without ever showing a raw traceback.
export class AnalysisError extends Error {
  constructor(message, { code = 'unknown_error', status = null, details = null } = {}) {
    super(message)
    this.name = 'AnalysisError'
    this.code = code
    this.status = status
    this.details = details
  }
}
