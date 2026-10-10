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
import traceback
from pathlib import Path

import numpy as np

from api.errors import (
    AnalysisAPIError,
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

API_ERROR_BY_CODE = {
    cls.error_code: cls
    for cls in (
        CorruptVideoError,
        FeatureExtractionFailedError,
        InferenceFailedError,
        AnalysisTimeoutError,
        MissingCheckpointAPIError,
    )
}

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
    extract_evidence_frames,
    require_file,
    load_feature_vector,
    load_single_stream_probability,
    _blink_status,
    lipsync_status_from_consistency,
    MissingCheckpointError,
    IncompatibleCheckpointError,
    MissingFeatureFileError,
    FeatureShapeError,
    DEFAULT_THRESHOLDS_PATH,
)
from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_CLASSIFIER_WEIGHTS,
    DEFAULT_FUSION_WEIGHTS,
    DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS,
    DEFAULT_VISUAL_CLASSIFIER_WEIGHTS,
)

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
        lower_reason = reason.lower()
        if ("no audio" in lower_reason or "does not contain any stream" in lower_reason
                or "does not contain an audio stream" in lower_reason
                or "audio: none" in lower_reason):
            logger.info("Audio analysis unavailable: no audio stream detected.")
            return {"status": "unavailable", "reason": "No audio stream detected"}
        raise FeatureExtractionFailedError(
            "Audio feature extraction failed.", details=f"audio/src/audio_stream.py subprocess: {reason}"
        ) from exc

    if not out_path.is_file():
        return {"status": "unavailable", "reason": "No audio stream detected"}

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
        result = analyze_lipsync(str(video_path), compute_windows=True)
        np.save(out_path, lipsync_result_to_vector(result))
    except Exception as exc:  # noqa: BLE001
        raise FeatureExtractionFailedError(
            "Lip-sync analysis failed.", details=f"visual/src/lipsync/lipsync_analyzer.py: {exc}"
        ) from exc
    return result


def _emit_progress(callback, stage, message):
    if callback is not None:
        callback({"stage": stage, "progress": None, "message": message})


def _run_no_audio_inference(job, blink_events, video_path, lipsync_result):
    """Run independent learned classifiers when no audio feature exists."""
    require_file(str(REPO_ROOT / DEFAULT_VISUAL_CLASSIFIER_WEIGHTS), "visual classifier checkpoint")
    require_file(str(REPO_ROOT / DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS), "semantic classifier checkpoint")
    visual = load_feature_vector(job.feature_path(job.visual_root), "visual")
    semantic = load_feature_vector(job.feature_path(job.semantic_root), "semantic")
    blink = load_feature_vector(job.feature_path(job.blink_root), "blink")
    visual_probability = load_single_stream_probability(
        visual, str(REPO_ROOT / DEFAULT_VISUAL_CLASSIFIER_WEIGHTS), 1280
    )
    semantic_probability = load_single_stream_probability(
        semantic, str(REPO_ROOT / DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS), 384
    )
    blink_score = float(blink[3])
    blink_status = _blink_status(blink_score)
    lipsync_score = None
    lipsync_status = "unavailable"
    window_evidence = "NOT AVAILABLE: no audio stream is available for lip-sync analysis."
    # A missing audio feature means no audio track to compare against;
    # do not run or report lip-sync analysis in this branch.

    evidence = {
        "evidence": ["Trained five-stream fusion was skipped because no audio feature was available."],
        "signals": [],
        "verdict": "UNAVAILABLE",
        "low_confidence": True,
        "evidence_consistency": "UNKNOWN",
        "modality_disagreement": "UNKNOWN",
        "review_recommended": True,
        "confidence_level": "unavailable",
    }
    return {
        "final_fake_probability": None,
        "final_real_probability": None,
        "prediction": "UNAVAILABLE",
        "visual_fake_probability": round(visual_probability, 4),
        "audio_fake_probability": None,
        "semantic_fake_probability": round(semantic_probability, 4),
        "blink_anomaly_score": round(blink_score, 4),
        "blink_status": blink_status,
        "lip_sync_mismatch_score": round(lipsync_score, 4) if lipsync_score is not None else None,
        "lip_sync_status": lipsync_status,
        "attention_summary": {},
        "modality_contributions": {"contributions": {}, "ablated_fake_probabilities": {}},
        "fusion_diagnostics": None,
        "trained_fusion_fake_probability": None,
        "final_real_probability": None,
        "evidence": evidence,
        "frame_evidence": None,
        "window_evidence": window_evidence,
        "_blink_events": blink_events,
        "_no_audio_reason": "audio feature unavailable",
        "_feature_health": {
            "visual": {"valid": True, "status": "valid"},
            "audio": {"valid": False, "status": "missing"},
            "semantic": {"valid": True, "status": "valid"},
            "blink": {"valid": True, "status": "valid"},
            "lipsync": {"valid": False, "status": "unavailable"},
        },
    }


def _run_raw_video_analysis(video_path: Path, job: Job, progress_callback=None) -> dict:
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

    _emit_progress(progress_callback, "probing_video", "Reading video metadata")
    meta = _probe_video(video_path)

    # Five per-stream extraction steps, each reusing the repo's own
    # existing extraction function - none reimplemented here.
    stage_started = time.monotonic()
    _emit_progress(progress_callback, "extracting_visual", "Extracting visual features")
    logger.info("[1/7] Extracting visual features...")
    _extract_visual(video_path, job)
    logger.info("[1/7] Extracting visual features completed in %.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    _emit_progress(progress_callback, "extracting_audio", "Extracting audio features")
    logger.info("[2/7] Extracting audio...")
    audio_evidence = _extract_audio(video_path, job)
    logger.info("[2/7] Extracting audio completed in %.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    _emit_progress(progress_callback, "extracting_semantic", "Extracting semantic features")
    logger.info("[3/7] Semantic embedding...")
    semantic_metadata = _extract_semantic(video_path, job)
    logger.info("[3/7] Semantic embedding completed in %.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    _emit_progress(progress_callback, "extracting_blink", "Analyzing eye-blink patterns")
    logger.info("[4/7] Blink analysis...")
    try:
        blink_events, blink_timeline = _extract_blink(video_path, job)
    except Exception as exc:  # supplementary analyzer cannot fail the learned result
        logger.warning("Supplementary blink analysis unavailable: %s", exc)
        blink_events, blink_timeline = [], []
    logger.info("[4/7] Blink analysis completed in %.3fs", time.monotonic() - stage_started)
    stage_started = time.monotonic()
    _emit_progress(progress_callback, "extracting_lipsync", "Analyzing lip synchronization")
    logger.info("[5/7] Lip-sync analysis...")
    try:
        lipsync_result = _extract_lipsync(video_path, job)
    except Exception as exc:  # supplementary analyzer cannot fail the learned result
        logger.warning("Supplementary lip-sync analysis unavailable: %s", exc)
        lipsync_result = None
    logger.info("[5/7] Lip-sync analysis completed in %.3fs", time.monotonic() - stage_started)

    feature_health_report = {
        name: feature_health(path, name)
        for name, path in {
            "visual": job.feature_path(job.visual_root),
            "audio": job.feature_path(job.audio_root),
            "semantic": job.feature_path(job.semantic_root),
        }.items()
    }

    try:
        stage_started = time.monotonic()
        _emit_progress(progress_callback, "running_classifiers", "Loading learned classifiers")
        logger.info("[6/7] Loading checkpoints...")
        _emit_progress(progress_callback, "running_fusion", "Combining visual, audio, and semantic results")
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
            fusion_weights=str(REPO_ROOT / DEFAULT_FUSION_WEIGHTS),
            thresholds_path=str(DEFAULT_THRESHOLDS_PATH),
            video_path=str(video_path),
            frame_output_dir=str(job.evidence_dir),
            blink_events=blink_events,
            precomputed_lipsync_result=lipsync_result,
        )
        logger.info("[6/7] Loading checkpoints completed in %.3fs", time.monotonic() - stage_started)
        logger.info("[7/7] Running fusion inference completed in %.3fs", time.monotonic() - stage_started)
        timings = result.get("_inference_timing_seconds", {})
        logger.info("Classifier inference completed in %.3fs", timings.get("classifier_inference", 0.0))
        if result.get("final_fake_probability") is not None:
            logger.info("Fusion inference completed in %.3fs", timings.get("fusion_inference", 0.0))
        logger.info("Evidence generation completed in %.3fs", timings.get("evidence_generation", 0.0))
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
                "Train the required stream classifiers and three-stream fusion model "
                "before running real inference."
            ),
        ) from exc
    except IncompatibleCheckpointError as exc:
        logger.error("Incompatible fusion checkpoint during raw-video analysis: %s", exc)
        raise InferenceFailedError(
            "The configured fusion checkpoint is incompatible with the active three-stream model.",
            details="See the backend logs for checkpoint metadata or architecture details.",
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
    result["_feature_health"] = result.get("_feature_health", feature_health_report)
    result["_semantic_metadata"] = semantic_metadata
    result["_audio_evidence"] = audio_evidence
    if result.get("_no_audio_reason"):
        result["_feature_health"]["audio"] = {
            "modality": "audio", "missing": True, "valid": False,
            "status": "missing", "shape": None, "dtype": None,
        }
        result["_audio_evidence"] = {"status": "unavailable", "reason": "No audio stream detected"}
    _emit_progress(progress_callback, "completed", "Analysis complete")

    return result, meta


def _analysis_worker(video_path, job_id, result_queue, progress_queue):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        result, meta = _run_raw_video_analysis(
            Path(video_path), Job(job_id),
            progress_callback=lambda event: progress_queue.put(event),
        )
        result_path = Job(job_id).dir / "analysis_result.json"
        logger.info("Preparing completed analysis for IPC: %s", result_path)
        with result_path.open("w", encoding="utf-8") as output:
            json.dump({"result": _to_ipc_native(result), "meta": _to_ipc_native(meta)}, output)
        logger.info("Completed analysis written for IPC: %s", result_path)
        result_queue.put(("ok", str(result_path)))
    except MissingCheckpointError as exc:
        result_queue.put(("missing_checkpoint", _worker_exception_payload(exc)))
    except (MissingFeatureFileError, FeatureShapeError) as exc:
        result_queue.put(("feature_error", _worker_exception_payload(exc)))
    except AnalysisAPIError as exc:
        payload = _worker_exception_payload(exc)
        payload["api_body"] = exc.to_body()
        result_queue.put(("api_error", payload))
    except Exception as exc:  # noqa: BLE001
        result_queue.put(("error", _worker_exception_payload(exc)))


def _worker_exception_payload(exc):
    """Serialize diagnostics for parent logging without sending them to clients."""
    return {
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "exception_repr": repr(exc),
        "traceback": traceback.format_exc(),
    }


def _log_worker_failure(status, payload):
    # Preserve the concise diagnostic that first identified the worker error.
    logger.error("Raw-video analysis worker failed: status=%r payload=%r", status, payload)
    if isinstance(payload, dict):
        logger.error(
            "Raw-video analysis worker failed: status=%r exception_type=%s "
            "exception_repr=%r message=%r\nWorker traceback:\n%s",
            status,
            payload.get("exception_type", "unknown"),
            payload.get("exception_repr", repr(payload)),
            payload.get("message"),
            payload.get("traceback", "Traceback unavailable."),
        )
    else:
        # Compatibility with a worker started from an older in-memory module.
        logger.error(
            "Raw-video analysis worker failed: status=%r exception_repr=%r\n"
            "Worker traceback unavailable.",
            status,
            repr(payload),
        )


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
        DEFAULT_FUSION_WEIGHTS,
    )
    return [path for path in paths if not (REPO_ROOT / path).is_file()]


def run_raw_video_analysis(video_path: Path, job: Job, progress_callback=None) -> dict:
    missing = _missing_checkpoint_paths()
    if missing:
        logger.error("Required checkpoints missing: %s", missing)
        raise MissingCheckpointAPIError(
            "Required trained model checkpoint not found.",
            details=(
                "See the backend server logs for the missing checkpoint paths."
            ),
        )

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    progress_queue = context.Queue()
    process = context.Process(
        target=_analysis_worker,
        args=(str(video_path), job.job_id, result_queue, progress_queue),
    )
    started_at = time.monotonic()
    logger.info("Raw-video analysis started at %.3f", started_at)
    process.start()
    deadline = started_at + ANALYSIS_TIMEOUT_SECONDS
    while process.is_alive() and time.monotonic() < deadline:
        process.join(0.2)
        while True:
            try:
                event = progress_queue.get_nowait()
                if progress_callback is not None:
                    progress_callback(event)
            except queue.Empty:
                break
    if process.is_alive():
        logger.error(
            "Raw-video analysis worker failed: status=%r exception_repr=%r "
            "worker_exit_code=%r; traceback unavailable because the timed-out worker is being terminated.",
            "timeout",
            AnalysisTimeoutError("analysis worker exceeded its deadline"),
            process.exitcode,
        )
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
        logger.error(
            "Raw-video analysis worker failed: status=%r exception_repr=%r "
            "worker_exit_code=%r; worker traceback unavailable because no payload was returned.",
            "no_result",
            RuntimeError("worker exited without a result payload"),
            process.exitcode,
        )
        raise InferenceFailedError(
            "Inference failed on the backend.",
            details="The analysis worker exited without returning a result.",
        )
    finally:
        result_queue.close()
        result_queue.join_thread()
        progress_queue.close()
        progress_queue.join_thread()

    logger.info("Raw-video analysis ended at %.3f", time.monotonic())
    if status == "ok":
        result_path = Path(payload)
        try:
            with result_path.open("r", encoding="utf-8") as input_file:
                envelope = json.load(input_file)
            return envelope["result"], envelope["meta"]
        finally:
            result_path.unlink(missing_ok=True)
    _log_worker_failure(status, payload)
    if status == "missing_checkpoint":
        raise MissingCheckpointAPIError(
            "Required trained model checkpoint not found.",
            details=(
                "Train the required stream classifiers and enhanced fusion model "
                "before running real inference."
            ),
        )
    if status == "feature_error":
        raise InferenceFailedError(
            "Inference failed due to an internal feature-preparation error.",
            details="See the backend server logs for the full error.",
        )
    if status == "api_error":
        api_body = payload.get("api_body", {}) if isinstance(payload, dict) else {}
        error_class = API_ERROR_BY_CODE.get(api_body.get("error"), InferenceFailedError)
        error_code = api_body.get("error")
        # Detailed extraction exceptions can contain machine paths or raw
        # FFmpeg output. Those details are available in the parent server log.
        safe_details = api_body.get("details") if error_code == "corrupt_video" else (
            "See the backend server logs for the full error."
            if error_class.status_code >= 500 else api_body.get("details")
        )
        raise error_class(
            api_body.get("message", "Inference failed on the backend."),
            details=safe_details,
        )

    raise InferenceFailedError(
        "Inference failed on the backend.",
        details="See the backend server logs for the full error.",
    )
