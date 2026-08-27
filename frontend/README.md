# Truthframe — Deepfake Detection Frontend

A standalone React frontend for the deepfake detection app. It calls a
backend at `POST /api/analyze` by default — see **Backend integration**
below for the exact contract and how to run against it, or against a
lightweight dev stand-in if you don't have the real backend running yet.

## Screen flow

```
Upload  →  Analysis (loading)  →  Results Dashboard              (or Error)
                                     ├── Overall Result
                                     ├── Modality Analysis (5 cards)
                                     ├── Evidence (Frames / Audio / Lip-Sync / Blink tabs)
                                     ├── Contextual Analysis
                                     └── Technical Details (attention + ablation)
```

## Getting started

```bash
npm install
npm run dev       # http://localhost:5173
```

By default this expects a backend at `http://localhost:8000` (proxied via
`/api`, see **Backend integration**). With no backend running yet, either:

- run the bundled dev stand-in: `npm run dev:mock-backend` (separate
  terminal) — implements the real contract, so the frontend's actual
  network/upload/error-handling code runs, just against canned responses
  instead of the real ML pipeline; or
- set `VITE_USE_MOCK_API=true` to skip the network entirely and render from
  the bundled mock JSON.

```bash
npm run build      # production build → dist/
npm run preview    # serve the production build locally
```

## Backend integration

`src/api/analyzeVideo.js` POSTs the uploaded file to `POST /api/analyze`
(`multipart/form-data`, field name `video`) and resolves with the backend's
JSON response — no filename-based or random result selection lives in the
production path anymore.

**`API_CONTRACT.md`** is the full spec: the exact response shape every
component reads from, the error-response shape, field-by-field mapping
notes from the pipeline's native names (`visual_fake_probability`,
`blink_anomaly_score`, `prediction`, etc.) to what the frontend expects, and
what happens when a piece of evidence (waveform, lip-sync windows, blink
timeline) genuinely isn't available for a given request — the UI shows an
explicit "___ evidence unavailable" message per tab rather than crashing or
fabricating a chart.

This pass could only build and verify the frontend side of the
Upload → API → Results integration — the actual backend repository
(`inference/full_pipeline.py`, `fusion/`, `evidence/`, etc.) wasn't
available in the environment that did this work. `API_CONTRACT.md` is
written so that building the real backend adapter is a matter of making
`run_full_inference()`'s output match that one document, in one
serializer — not touching any React component.

**Local dev (two terminals), once the real backend exists:**

```bash
# Terminal 1 — backend (adjust to your actual entrypoint/venv)
cd <backend-repo>
<activate venv>
uvicorn api.server:app --port 8000     # or however it's actually run

# Terminal 2 — frontend
cd <frontend-repo>
npm install
npm run dev
```

Then open http://localhost:5173, upload a video, click Analyze.

**Env vars** (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | unset | Absolute backend URL for builds where frontend/backend are on different origins. Leave unset for local dev — requests go to relative `/api/...`. |
| `BACKEND_URL` | `http://localhost:8000` | Only used by `npm run dev`'s proxy target (`vite.config.js`), not the built app. |
| `VITE_USE_MOCK_API` | `false` | `true` bypasses the network and resolves the bundled mock JSON — UI-only development. |

**Dev-only mock backend:** `scripts/dev-mock-server.mjs` implements
`GET /api/health` and `POST /api/analyze` per `API_CONTRACT.md`, returning
the bundled fixtures — it runs no model. Useful for testing the frontend's
real HTTP/error-handling path without the ML backend:

```bash
npm run dev:mock-backend                          # alternates fake/real
MOCK_FAIL=missing_checkpoint npm run dev:mock-backend   # force an error state
```

Delete this file once the real backend implements the endpoint — it's a
stand-in, not part of the shipped app.

## Project layout

```
API_CONTRACT.md               ← the request/response contract — read this first
src/
  api/
    client.js                 ← API_BASE_URL / USE_MOCK_API / evidence-URL resolution
    analyzeVideo.js            ← real fetch() to POST /api/analyze
  mock/
    mockResult.fake.json      ← fixture, deepfake verdict (used by VITE_USE_MOCK_API + dev-mock-server)
    mockResult.real.json      ← fixture, real verdict
  screens/
    UploadScreen.jsx
    AnalysisScreen.jsx
    ResultsDashboard.jsx
    ErrorScreen.jsx           ← shown when analyzeVideo() rejects
  components/
    OverallResultPanel.jsx
    ModalityCard.jsx
    EvidenceViewer.jsx        ← tabs: Frames / Audio / Lip-Sync / Blink, each with an empty state
    FrameViewer.jsx
    AudioTimeline.jsx
    LipSyncTimeline.jsx
    BlinkTimeline.jsx
    ContextualAnalysis.jsx
    TechnicalDetails.jsx
    modalityMeta.js           ← single source of truth for labels/colors/kind
    ui/                       ← Badge, Meter, InfoTip, EmptyState
  styles/
    tokens.css                ← color/spacing/radius/shadow design tokens
    base.css
    components.css
scripts/
  generate-mock.mjs           ← regenerate the two mock JSON fixtures
  dev-mock-server.mjs         ← contract-shaped stand-in backend for local testing (not production)
```

## Why some scores are "anomaly/mismatch score" and not a probability

This mirrors a real distinction in the model repo, not just a UI choice:
`visual`, `audio`, and `semantic` are outputs of trained, independently
evaluated classifiers, so they're shown as **fake probability**. `eye_blink`
and `lip_sync` are rule-based signals (blink rate/duration/EAR pattern;
mouth-motion-vs-audio correlation) — they're real fusion **inputs**, not an
afterthought, but they are not yet calibrated probabilities. The UI labels
them **anomaly score** / **mismatch score** everywhere (cards, evidence tabs,
tooltips) to keep that honest. `src/components/modalityMeta.js` is the single
place this labeling is defined — change it there if the calibration story
changes later, and it'll stay consistent across the whole app.

The same care applies to two more panels in Technical Details:

- **Attention weights** are descriptive (where the fusion model's attention
  landed), not a causal importance ranking.
- **Modality contribution** is ablation-based (how much the fake probability
  drops when a modality is zeroed out) — a heuristic, not ground-truth
  attribution.

Both have an inline `?` tooltip carrying that caveat verbatim.

## Design notes

Light theme, Inter type, a small custom design-token system in
`src/styles/tokens.css` (no UI framework/Tailwind dependency — easy to swap
colors org-wide by editing one file). Charts (waveform, blink EAR line,
lip-sync mismatch bars) are hand-rolled inline SVG — no charting library
dependency, and they hover-highlight to show exact values.
