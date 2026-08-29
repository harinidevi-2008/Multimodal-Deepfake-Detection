"""
Raw-video -> run_full_inference() orchestration.

inference/full_pipeline.py's own docstring is explicit about its scope:
it runs on a sample whose five modality features have ALREADY been
precomputed to .npy files - it does NOT do raw feature extraction on an
arbitrary new video. That is exactly the gap this module fills, by
calling each stream's EXISTING per-video extraction function (never a
reimplementation) to produce those five .npy files into a job-scoped
temporary feature root, then calling run_full_inference() itself
exactly like every other caller in this repo does.

Reused, not reimplemented:
    visual   -> visual/src/process_video.py:process_video()
    audio    -> audio/src/audio_stream.py:process_video()  (isolated in
                a subprocess - see api/runners/run_audio_extract.py for why)
    semantic -> semantic/src/semantic_stream.py:SemanticStream.extract_features()
    blink    -> visual/src/eye_blink/blink_analyzer.py:analyze_blinks()
                + visual/src/blink_lipsync_precompute.py:blink_result_to_vector()
    lipsync  -> visual/src/lipsync/lipsync_analyzer.py:analyze_lipsync()
                + blink_lipsync_precompute.py:lipsync_result_to_vector()
    fusion   -> inference/full_pipeline.py:run_full_inference()
                (also internally re-derives blink events and windowed
                lip-sync evidence from video_path - see its own
                docstring; not duplicated here)
"""

import logging
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from api.errors import (
    CorruptVideoError,
    FeatureExtractionFailedError,
    InferenceFailedError,
    AnalysisTimeoutError,
    MissingCheckpointAPIError,
)
from api.jobs import RELATIVE_FEATURE_PATH, Job

logger = logging.getLogger("deepfake_api")

REPO_ROOT = Path(__file__).resolve().parent.parent
VISUAL_SRC = REPO_ROOT / "visual" / "src"
SEMANTIC_SRC = REPO_ROOT / "semantic" / "src"
INFERENCE_DIR = REPO_ROOT / "inference"
AUDIO_RUNNER = Path(__file__).resolve().parent / "runners" / "run_audio_extract.py"

AUDIO_EXTRACT_TIMEOUT_SECONDS = 300
ANALYSIS_TIMEOUT_SECONDS = int(os.environ.get("DFD_ANALYSIS_TIMEOUT_SECONDS", "900"))

# ---------------------------------------------------------------------
# sys.path bootstrap (main process only - audio/src is deliberately
# NEVER added here; see api/runners/run_audio_extract.py's docstring
# for the visual/audio config.py module-name collision this avoids).
# ---------------------------------------------------------------------
for _p in (str(INFERENCE_DIR), str(VISUAL_SRC), str(SEMANTIC_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402

# Importing full_pipeline has the side effect of also putting fusion/,
# classifiers/, evidence/, visual/src/lipsync, visual/src/eye_blink on
# sys.path (see its own module docstring/top) - matching exactly how
# inference/test_full_pipeline_smoke.py already relies on the same
# side effect, rather than duplicating that sys.path setup here.
import full_pipeline  # noqa: E402
from full_pipeline import (  # noqa: E402
    run_full_inference,
    MissingCheckpointError,
    MissingFeatureFileError,
    FeatureShapeError,
)
from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_CLASSIFIER_WEIGHTS,
    DEFAULT_ENHANCED_FUSION_WEIGHTS,
    DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS,
    DEFAULT_VISUAL_CLASSIFIER_WEIGHTS,
)
from feature_normalization import DEFAULT_NORMALIZATION_PATH  # noqa: E402

from process_video import process_video as extract_visual_feature  # noqa: E402
from blink_analyzer import (  # noqa: E402 (visual/src/eye_blink, via full_pipeline's sys.path insert)
    analyze_blinks,
    compute_ear_series,
    detect_blinks,
)
from lipsync_analyzer import analyze_lipsync  # noqa: E402 (visual/src/lipsync, via full_pipeline's sys.path insert)
from blink_lipsync_precompute import (  # noqa: E402
    blink_result_to_vector,
    lipsync_result_to_vector,
)

# ---------------------------------------------------------------------
# Lazy-loaded SemanticStream singleton (Whisper + Sentence-BERT are
# expensive to load - matches the lazy-load convention already used
# throughout this repo for EfficientNet/MTCNN/Wav2Vec2, see
# audio/src/audio_stream.py's _get_wav2vec2()).
# ---------------------------------------------------------------------
_semantic_stream = None


def get_semantic_stream():
    global _semantic_stream
    if _semantic_stream is None:
        from semantic_stream import SemanticStream  # noqa: E402

        logger.info("Loading SemanticStream (Whisper + Sentence-BERT) - first request only...")
        _semantic_stream = SemanticStream()
    return _semantic_stream


def _probe_video(video_path: Path) -> dict:
    """Open the upload once with OpenCV to confirm it's a readable video
    and to read its own genuine fps/frame-count/duration - used both to
    reject a corrupt upload early and to compute timestamp_seconds for
    evidence frames later (never a fabricated/assumed fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise CorruptVideoError(
            "Could not open the uploaded file as a video.",
            details="OpenCV VideoCapture failed to open the file - it may not be a valid video container.",
        )
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()
    if total_frames <= 0:
        raise CorruptVideoError(
            "The uploaded video appears to contain no readable frames.",
            details=f"OpenCV reported frame count={total_frames}.",
        )
    duration_seconds = (total_frames / fps) if fps else None
    return {"fps": fps, "total_frames": total_frames, "duration_seconds": duration_seconds}


def _extract_visual(video_path: Path, job: Job):
    out_path = job.feature_path(job.visual_root)
    try:
        feature = extract_visual_feature(str(video_path), str(out_path))
    except Exception as exc:  # noqa: BLE001 - report, don't fabricate a feature
        raise FeatureExtractionFailedError(
            "Visual feature extraction failed.", details=f"visual/src/process_video.py: {exc}"
        ) from exc
    if feature is None:
        raise FeatureExtractionFailedError(
            "No face was detected in any sampled frame of the uploaded video.",
            details="visual/src/process_video.py returned None (see its own no-faces-detected log line).",
        )


def _extract_audio(video_path: Path, job: Job):
    out_path = job.feature_path(job.audio_root)
    try:
        subprocess.run(
            [sys.executable, str(AUDIO_RUNNER), str(video_path), str(out_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=AUDIO_EXTRACT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise FeatureExtractionFailedError(
            "Audio feature extraction timed out.",
            details=f"audio extraction exceeded {AUDIO_EXTRACT_TIMEOUT_SECONDS} seconds.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        reason = (exc.stderr or exc.stdout or "unknown error").strip()[-500:]
        raise FeatureExtractionFailedError(
            "Audio feature extraction failed.", details=f"audio/src/audio_stream.py subprocess: {reason}"
        ) from exc

    return _extract_audio_evidence(video_path)


def _extract_audio_evidence(video_path: Path):
    """Return a compact real RMS envelope; it does not identify anomalies."""
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            wav_path = Path(temp_file.name)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-ac", "1", "-ar", "16000", "-vn", str(wav_path)],
            capture_output=True, text=True, check=True, timeout=AUDIO_EXTRACT_TIMEOUT_SECONDS,
        )
        import librosa
        waveform, sample_rate = librosa.load(str(wav_path), sr=16000)
        if waveform.size == 0:
            return {"status": "unavailable"}
        point_count = min(256, max(1, waveform.size // max(1, sample_rate // 20)))
        chunks = np.array_split(np.abs(waveform), point_count)
        envelope = np.asarray([float(np.sqrt(np.mean(chunk ** 2))) for chunk in chunks], dtype=np.float32)
        peak = float(envelope.max())
        if peak > 0:
            envelope /= peak
        return {
            "status": "global_only",
            "sample_rate_hz": sample_rate,
            "duration_seconds": round(float(waveform.size / sample_rate), 3),
            "envelope": [round(float(value), 5) for value in envelope],
            "timestamps_seconds": [round(float(i * waveform.size / sample_rate / len(envelope)), 3)
                                   for i in range(len(envelope))],
            "localized_windows": [],
            "localization_note": "Audio classifier evidence is global only; no timestamped suspicious audio intervals were produced.",
        }
    except Exception as exc:  # noqa: BLE001 - waveform is optional evidence
        logger.warning("Audio envelope unavailable: %s", exc)
        return {"status": "unavailable"}
    finally:
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)


def feature_health(path, modality):
    """Summarize one generated feature without changing inference behavior."""
    path = Path(path)
    health = {"modality": modality, "path": str(path.resolve()), "missing": not path.is_file()}
    if health["missing"]:
        health.update({"valid": False, "status": "missing", "shape": None, "dtype": None,
                       "min": None, "max": None, "mean": None, "std": None,
                       "finite": False, "zero_fraction": None})
        return health
    try:
        array = np.asarray(np.load(path))
        finite = bool(np.isfinite(array).all())
        zero_fraction = float(np.mean(array == 0)) if array.size else 1.0
        near_zero = bool(array.size == 0 or np.max(np.abs(array)) <= 1e-8)
        valid = bool(array.size and finite and not near_zero)
        status = "zero_embedding" if modality == "semantic" and near_zero else ("degraded" if not valid else "valid")
        health.update({"valid": valid, "status": status, "shape": list(array.shape),
                       "dtype": str(array.dtype), "min": float(np.min(array)) if array.size else None,
                       "max": float(np.max(array)) if array.size else None,
                       "mean": float(np.mean(array)) if array.size else None,
                       "std": float(np.std(array)) if array.size else None,
                       "finite": finite, "zero_fraction": zero_fraction})
        logger.info("Feature health %s: shape=%s dtype=%s min=%s max=%s mean=%s std=%s finite=%s zero_fraction=%.4f status=%s",
                    modality, health["shape"], health["dtype"], health["min"], health["max"],
                    health["mean"], health["std"], finite, zero_fraction, status)
        if not valid:
            logger.warning("Feature health warning for %s: %s", modality, status)
        return health
    except Exception as exc:  # noqa: BLE001
        logger.warning("Feature health failed for %s: %s", modality, exc)
        health.update({"valid": False, "status": "invalid", "error": str(exc)})
        return health


def _extract_semantic(video_path: Path, job: Job):
    out_path = job.feature_path(job.semantic_root)
    try:
        stream = get_semantic_stream()
        result = stream.extract_features(str(video_path))
        embedding = result["embedding_384"].detach().cpu().numpy().astype(np.float32)
        np.save(out_path, embedding)
        return {
            "transcript": result.get("transcript", ""),
            "confidence": result.get("confidence"),
            "reliable": result.get("reliable", False),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
        }
    except Exception as exc:  # noqa: BLE001
        raise FeatureExtractionFailedError(
            "Semantic feature extraction failed.", details=f"semantic/src/semantic_stream.py: {exc}"
        ) from exc


def _extract_blink(video_path: Path, job: Job):
    """
    Two real calls into blink_analyzer.py, not a reimplementation of
    either: analyze_blinks() for the aggregate 4-d feature vector (same
    function visual/src/blink_lipsync_precompute.py uses when
    precomputing this feature for the training set), and a direct
    compute_ear_series()+detect_blinks() pass - the same two functions
    analyze_blinks() itself calls internally - to get the per-frame EAR
    series and blink-event boundaries needed for evidence.blink_timeline
    and to hand genuine blink_events to run_full_inference() (so it
    doesn't need to re-derive them itself via obtain_blink_events()).

    Returns (blink_events, blink_timeline):
        blink_events: list[(start_frame, end_frame)]
        blink_timeline: list[{"timestamp_seconds", "ear_value", "is_blink"}] -
            one entry per frame where a face/landmarks were actually
            found; frames with no detected face are skipped rather than
            filled with a fabricated EAR value.
    """
    out_path = job.feature_path(job.blink_root)
    try:
        vector_result = analyze_blinks(str(video_path))
        np.save(out_path, blink_result_to_vector(vector_result))

        ear_data = compute_ear_series(str(video_path))
        ear_values = ear_data["ear_values"]
        fps = ear_data["fps"]
        blink_events = detect_blinks(ear_values)

        blink_timeline = []
        for i, ear in enumerate(ear_values):
            if ear is None:
                continue
            is_blink = any(start <= i <= end for start, end in blink_events)
            blink_timeline.append({
                "timestamp_seconds": round(i / fps, 3) if fps else None,
                "ear_value": round(float(ear), 4),
                "is_blink": is_blink,
            })
    except Exception as exc:  # noqa: BLE001
        raise FeatureExtractionFailedError(
            "Eye-blink analysis failed.", details=f"visual/src/eye_blink/blink_analyzer.py: {exc}"
        ) from exc

    return blink_events, blink_timeline


def _extract_lipsync(video_path: Path, job: Job):
    out_path = job.feature_path(job.lipsync_root)
    try:
        result = analyze_lipsync(str(video_path))
        np.save(out_path, lipsync_result_to_vector(result))
    except Exception as exc:  # noqa: BLE001
        raise FeatureExtractionFailedError(
            "Lip-sync analysis failed.", details=f"visual/src/lipsync/lipsync_analyzer.py: {exc}"
        ) from exc


def _run_raw_video_analysis(video_path: Path, job: Job) -> dict:
    """
    Full raw-video pipeline for one uploaded file, already saved at
    video_path under job.upload_dir.

    Returns (result, meta):
        result: run_full_inference()'s native result dict, plus one
            extra key this module adds (_blink_timeline - see
            _extract_blink()'s docstring). api/serializer.py converts
            this into the frontend's API_CONTRACT.md shape.
        meta: {"fps", "total_frames", "duration_seconds"} read directly
            from the upload via OpenCV (_probe_video) - needed to
            compute evidence.frames[].timestamp_seconds and
            video_duration_seconds without re-opening the video again
            in the serializer.

    Raises an api.errors.AnalysisAPIError subclass on any failure -
    server.py's exception handlers (installed via
    api.errors.install_error_handlers) turn these into the structured
    JSON error response, and nothing else needs to catch them.
    """
    video_path = Path(video_path)

    meta = _probe_video(video_path)

    # Five per-stream extraction steps, each reusing the repo's own
    # existing extraction function - none reimplemented here.
    stage_started = time.monotonic()
    logger.info("[1/7] Extracting visual features...")
    _extract_visual(video_path, job)
    logger.info("[1/7] Extracting visual features completed in %.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    logger.info("[2/7] Extracting audio...")
    audio_evidence = _extract_audio(video_path, job)
    logger.info("[2/7] Extracting audio completed in %.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    logger.info("[3/7] Semantic embedding...")
    semantic_metadata = _extract_semantic(video_path, job)
    logger.info("[3/7] Semantic embedding completed in %.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    logger.info("[4/7] Blink analysis...")
    blink_events, blink_timeline = _extract_blink(video_path, job)
    logger.info("[4/7] Blink analysis completed in %.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    logger.info("[5/7] Lip-sync analysis...")
    _extract_lipsync(video_path, job)
    logger.info("[5/7] Lip-sync analysis completed in %.3fs", time.monotonic() - stage_started)

    feature_health_report = {
        name: feature_health(path, name)
        for name, path in {
            "visual": job.feature_path(job.visual_root),
            "audio": job.feature_path(job.audio_root),
            "semantic": job.feature_path(job.semantic_root),
            "blink": job.feature_path(job.blink_root),
            "lipsync": job.feature_path(job.lipsync_root),
        }.items()
    }

    try:
        stage_started = time.monotonic()
        logger.info("[6/7] Loading checkpoints...")
        logger.info("[7/7] Running fusion inference...")
        result = run_full_inference(
            str(RELATIVE_FEATURE_PATH),
            visual_root=str(job.visual_root),
            audio_root=str(job.audio_root),
            semantic_root=str(job.semantic_root),
            blink_root=str(job.blink_root),
            lipsync_root=str(job.lipsync_root),
            visual_classifier_weights=str(REPO_ROOT / DEFAULT_VISUAL_CLASSIFIER_WEIGHTS),
            audio_classifier_weights=str(REPO_ROOT / DEFAULT_AUDIO_CLASSIFIER_WEIGHTS),
            semantic_classifier_weights=str(REPO_ROOT / DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS),
            enhanced_fusion_weights=str(REPO_ROOT / DEFAULT_ENHANCED_FUSION_WEIGHTS),
            normalization_path=(
                str(DEFAULT_NORMALIZATION_PATH) if Path(DEFAULT_NORMALIZATION_PATH).exists() else None
            ),
            video_path=str(video_path),
            frame_output_dir=str(job.evidence_dir),
            blink_events=blink_events,
        )
        logger.info("[6/7] Loading checkpoints completed in %.3fs", time.monotonic() - stage_started)
        logger.info("[7/7] Running fusion inference completed in %.3fs", time.monotonic() - stage_started)
    except MissingCheckpointError as exc:
        # Expected, correct, by-design: real training has not yet
        # produced these checkpoints. Never faked, never downgraded to
        # a generic 500 - this is specifically "not ready yet", not
        # "broken". str(exc) (from full_pipeline.require_file()) names
        # the exact missing checkpoint by its absolute server path -
        # genuinely useful for debugging, but never safe to hand the
        # client, so it's logged in full server-side only and the
        # client gets a generic, honest, path-free explanation instead.
        logger.error("MissingCheckpointError during raw-video analysis: %s", exc)
        raise MissingCheckpointAPIError(
            "The trained models required for analysis are not available yet.",
            details=(
                "Train the required stream classifiers and enhanced fusion model "
                "before running real inference."
            ),
        ) from exc
    except (MissingFeatureFileError, FeatureShapeError) as exc:
        # Would indicate a bug in the five extraction steps above (they
        # should always produce all five correctly-shaped files before
        # this point is reached) - report as inference_failed rather
        # than silently retrying or fabricating a result. Same
        # path-leak concern as above: the full message (which names an
        # absolute server-side feature-file path) is logged, not sent.
        logger.error("Feature-preparation error during raw-video analysis: %s", exc)
        raise InferenceFailedError(
            "Inference failed due to an internal feature-preparation error.",
            details="See the backend server logs for the full error.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error during raw-video analysis: %s", exc)
        raise InferenceFailedError(
            "Inference failed on the backend.",
            details="See the backend server logs for the full error.",
        ) from exc

    # Genuine per-frame blink data, gathered above via the same
    # compute_ear_series()/detect_blinks() functions blink_analyzer.py
    # itself uses - attached here (not returned by run_full_inference,
    # which only returns the aggregate score) so api/serializer.py can
    # build evidence.blink_timeline without recomputing it a third time.
    result["_blink_timeline"] = blink_timeline
    result["_blink_events"] = blink_events
    result["_feature_health"] = feature_health_report
    result["_semantic_metadata"] = semantic_metadata
    result["_audio_evidence"] = audio_evidence

    return result, meta


def _analysis_worker(video_path, job_id, result_queue):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        result, meta = _run_raw_video_analysis(Path(video_path), Job(job_id))
        result_path = Job(job_id).dir / "analysis_result.json"
        logger.info("Preparing completed analysis for IPC: %s", result_path)
        with result_path.open("w", encoding="utf-8") as output:
            json.dump({"result": _to_ipc_native(result), "meta": _to_ipc_native(meta)}, output)
        logger.info("Completed analysis written for IPC: %s", result_path)
        result_queue.put(("ok", str(result_path)))
    except MissingCheckpointError as exc:
        result_queue.put(("missing_checkpoint", str(exc)))
    except (MissingFeatureFileError, FeatureShapeError) as exc:
        result_queue.put(("feature_error", str(exc)))
    except Exception as exc:  # noqa: BLE001
        result_queue.put(("error", repr(exc)))


def _to_ipc_native(value):
    """Convert ML result values to small JSON-native structures before IPC."""
    if isinstance(value, dict):
        return {str(key): _to_ipc_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_ipc_native(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().tolist()
    return value


def _missing_checkpoint_paths():
    paths = (
        DEFAULT_VISUAL_CLASSIFIER_WEIGHTS,
        DEFAULT_AUDIO_CLASSIFIER_WEIGHTS,
        DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS,
        DEFAULT_ENHANCED_FUSION_WEIGHTS,
    )
    return [path for path in paths if not (REPO_ROOT / path).is_file()]


def run_raw_video_analysis(video_path: Path, job: Job) -> dict:
    missing = _missing_checkpoint_paths()
    if missing:
        logger.error("Required checkpoints missing: %s", missing)
        raise MissingCheckpointAPIError(
            "Required trained model checkpoint not found.", details=missing
        )

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_analysis_worker,
        args=(str(video_path), job.job_id, result_queue),
    )
    started_at = time.monotonic()
    logger.info("Raw-video analysis started at %.3f", started_at)
    process.start()
    process.join(ANALYSIS_TIMEOUT_SECONDS)
    if process.is_alive():
        logger.error("Raw-video analysis timed out after %ss; terminating worker", ANALYSIS_TIMEOUT_SECONDS)
        process.terminate()
        process.join(5)
        (job.dir / "analysis_result.json").unlink(missing_ok=True)
        raise AnalysisTimeoutError(
            "Analysis timed out while running the backend pipeline.",
            details=f"The request exceeded the {ANALYSIS_TIMEOUT_SECONDS}-second analysis limit.",
        )

    try:
        status, payload = result_queue.get(timeout=2)
    except queue.Empty:
        raise InferenceFailedError(
            "Inference failed on the backend.",
            details="The analysis worker exited without returning a result.",
        )
    finally:
        result_queue.close()
        result_queue.join_thread()

    logger.info("Raw-video analysis ended at %.3f", time.monotonic())
    if status == "ok":
        result_path = Path(payload)
        try:
            with result_path.open("r", encoding="utf-8") as input_file:
                envelope = json.load(input_file)
            return envelope["result"], envelope["meta"]
        finally:
            result_path.unlink(missing_ok=True)
    if status == "missing_checkpoint":
        raise MissingCheckpointAPIError(
            "Required trained model checkpoint not found.", details=[payload]
        )
    if status == "feature_error":
        raise InferenceFailedError(
            "Inference failed due to an internal feature-preparation error.",
            details="See the backend server logs for the full error.",
        )
    raise InferenceFailedError(
        "Inference failed on the backend.",
        details="See the backend server logs for the full error.",
    )
