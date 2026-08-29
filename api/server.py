"""
FastAPI application - the thin adapter layer around the existing
pipeline. This file owns HTTP concerns only (routing, multipart
handling, CORS, job lifecycle, evidence file serving); all pipeline
orchestration lives in api/pipeline_adapter.py and all response
shaping lives in api/serializer.py, per the integration spec's "keep
the frontend contract and pipeline-calling code in as few places as
possible" instruction.

Run (from the repo root, inside the project's venv):
    uvicorn api.server:app --host 0.0.0.0 --port 8000
"""

import datetime
import logging
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.errors import (
    FileTooLargeError,
    MissingVideoError,
    NotFoundError,
    UnsupportedMediaTypeError,
    install_error_handlers,
)
from api.jobs import JOBS_ROOT, new_job, sweep_old_jobs
from api.pipeline_adapter import run_raw_video_analysis
from api.serializer import serialize_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("deepfake_api")

# Matches audio/src/config.py's SUPPORTED_EXTENSIONS - the audio stream
# (and therefore the whole 5-modal pipeline, which requires all five
# features) is not verified to work on any container format beyond
# these three, so this API doesn't advertise support for more.
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

# Narrow CORS: the frontend's own dev server origin(s) only - never "*"
# for an endpoint that accepts file uploads and serves files back.
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app = FastAPI(title="Multimodal Deepfake Detection API")
install_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
async def health():
    # Deliberately does not load or touch any model - a liveness check
    # only, per API_CONTRACT.md.
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(None)):
    if video is None or not video.filename:
        raise MissingVideoError("No video file was provided.")

    suffix = Path(video.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise UnsupportedMediaTypeError(
            "That file type is not supported - please upload a video.",
            details=f"Got '{suffix or '(no extension)'}', expected one of {sorted(ALLOWED_EXTENSIONS)}.",
        )

    sweep_old_jobs()
    job = new_job()
    upload_path = job.upload_dir / f"upload{suffix}"

    try:
        size = 0
        with open(upload_path, "wb") as out:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise FileTooLargeError(
                        "The uploaded file is too large.",
                        details=f"Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                out.write(chunk)
        if size == 0:
            raise MissingVideoError("The uploaded video file was empty.")

        start = time.monotonic()
        result, meta = run_raw_video_analysis(upload_path, job)
        processing_time_seconds = time.monotonic() - start

        analyzed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        return serialize_result(
            result,
            meta,
            job_id=job.job_id,
            video_filename=video.filename,
            processing_time_seconds=processing_time_seconds,
            analyzed_at=analyzed_at,
        )
    finally:
        # Keep the upload until the job directory expires so the UI can
        # replay genuine source audio/video for evidence review. Old
        # jobs are still cleaned up by sweep_old_jobs().
        pass


@app.get("/api/evidence/{job_id}/{filename}")
async def get_evidence(job_id: str, filename: str):
    # Path-traversal protection: reject anything that isn't a bare
    # filename (no "..", no "/", no leading "."), then resolve and
    # confirm the final path is still inside this job's own evidence
    # directory before serving it - two independent checks rather than
    # relying on string-matching alone.
    if "/" in filename or "\\" in filename or filename in ("..", ".") or filename.startswith("."):
        raise NotFoundError("No such evidence file.")
    if not job_id.isalnum():
        raise NotFoundError("No such job.")

    job_dir = (JOBS_ROOT / job_id / "evidence").resolve()
    candidate = (job_dir / filename).resolve()

    if job_dir not in candidate.parents and candidate != job_dir:
        raise NotFoundError("No such evidence file.")
    if not candidate.is_file():
        raise NotFoundError("No such evidence file.")

    return FileResponse(str(candidate), media_type="image/jpeg")


@app.get("/api/media/{job_id}")
async def get_media(job_id: str):
    if not job_id.isalnum():
        raise NotFoundError("No such job.")

    upload_dir = (JOBS_ROOT / job_id / "upload").resolve()
    if not upload_dir.is_dir():
        raise NotFoundError("No such media file.")

    candidates = [path for path in upload_dir.iterdir() if path.is_file()]
    if len(candidates) != 1:
        raise NotFoundError("No such media file.")

    media_type_by_suffix = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/x-msvideo",
    }
    media_path = candidates[0]
    return FileResponse(
        str(media_path),
        media_type=media_type_by_suffix.get(media_path.suffix.lower(), "application/octet-stream"),
    )
