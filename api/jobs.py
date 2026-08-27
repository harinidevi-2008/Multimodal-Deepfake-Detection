"""
Job-scoped temporary file layout.

Every /api/analyze request gets its own job_id (uuid4) and its own
directory tree under runtime/jobs/<job_id>/ - so concurrent requests
never share (or race on) feature files, and job-specific feature roots
can be passed straight into run_full_inference() as visual_root=/
audio_root=/etc. overrides without touching the real dataset feature
directories (visual/data/features_aligned, audio/data/features, ...)
at all.

Layout:
    runtime/jobs/<job_id>/
        upload/<original_filename>          - the uploaded video, deleted after the request
        features/visual/<REL>.npy
        features/audio/<REL>.npy
        features/semantic/<REL>.npy
        features/blink/<REL>.npy
        features/lipsync/<REL>.npy
        evidence/                            - frame_evidence.py's extracted .jpg frames, served
                                                back to the client via GET /api/evidence/{job_id}/{filename}

REL is always the fixed relative path RELATIVE_FEATURE_PATH below - one
sample per job, so there's no dataset-style nested category/id
structure to preserve; it exists only because run_full_inference()'s
signature is (relative_path, visual_root, ...) and joins them itself.
"""

import shutil
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
JOBS_ROOT = REPO_ROOT / "runtime" / "jobs"

# The single fixed "relative path" every job's precomputed features are
# saved under, within that job's own feature roots. Any name is fine -
# run_full_inference() only ever sees this one job's own directories -
# but keeping it descriptive helps if a job dir is inspected manually.
RELATIVE_FEATURE_PATH = Path("sample.npy")

# Jobs older than this are eligible for cleanup on the next /api/analyze
# call (a simple, dependency-free policy - no background scheduler).
JOB_MAX_AGE_SECONDS = 2 * 60 * 60  # 2 hours


class Job:
    def __init__(self, job_id):
        self.job_id = job_id
        self.dir = JOBS_ROOT / job_id
        self.upload_dir = self.dir / "upload"
        self.features_dir = self.dir / "features"
        self.evidence_dir = self.dir / "evidence"

        self.visual_root = self.features_dir / "visual"
        self.audio_root = self.features_dir / "audio"
        self.semantic_root = self.features_dir / "semantic"
        self.blink_root = self.features_dir / "blink"
        self.lipsync_root = self.features_dir / "lipsync"

    def create_dirs(self):
        for d in (
            self.upload_dir,
            self.visual_root,
            self.audio_root,
            self.semantic_root,
            self.blink_root,
            self.lipsync_root,
            self.evidence_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def feature_path(self, root: Path) -> Path:
        p = root / RELATIVE_FEATURE_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def cleanup_upload(self):
        """Delete the uploaded video once analysis is done (success or
        failure) - only the small evidence frames/features are worth
        keeping around for the evidence-serving route; the original
        upload is not re-read after run_full_inference() returns."""
        shutil.rmtree(self.upload_dir, ignore_errors=True)


def new_job() -> Job:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    job = Job(uuid.uuid4().hex)
    job.create_dirs()
    return job


def sweep_old_jobs(max_age_seconds=JOB_MAX_AGE_SECONDS):
    """Best-effort cleanup of job directories older than max_age_seconds.
    Called opportunistically (e.g. once per /api/analyze call) rather
    than on a background timer, to avoid adding a scheduler dependency
    for what is currently a low-traffic internal API. Never raises -
    cleanup failing must not fail the request that triggered it."""
    if not JOBS_ROOT.exists():
        return
    now = time.time()
    try:
        for entry in JOBS_ROOT.iterdir():
            try:
                if entry.is_dir() and (now - entry.stat().st_mtime) > max_age_seconds:
                    shutil.rmtree(entry, ignore_errors=True)
            except OSError:
                continue
    except OSError:
        pass
