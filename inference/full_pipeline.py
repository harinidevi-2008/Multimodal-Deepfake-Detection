"""
Application-facing inference orchestration layer.

This does NOT reimplement anything - it wires together modules that
already exist and already work on their own:
  - classifiers/single_stream_classifier.py  (visual/audio/semantic P(fake), each its own trained model)
  - fusion/fusion_model.py                  (final three-stream P(fake))
  - fusion/fusion_model.py                   (the active three-stream final score)
  - evidence/evidence_builder.py             (human-readable evidence reasons)
  - evidence/frame_evidence.py               (frame selection metadata - never "detected_artifact")
  - fusion/evaluate_blink_lipsync.py's BLINK_ANOMALY_THRESHOLD and
    visual/src/lipsync/lipsync_analyzer.py's lipsync_status_from_consistency()
    (the SAME rule-based status boundaries used everywhere else in this repo)

IMPORTANT - one requirement from the correction pass, reiterated here
because it is easy to get wrong: visual_fake_probability,
audio_fake_probability, and semantic_fake_probability MUST come from
each stream's own separately trained SingleStreamClassifier. They are
NEVER derived from the fusion model's internal projections or
attention - the fusion model was trained end-to-end and its internal
representations are not equivalent to, and must not be presented as,
an independent single-stream classifier's output.

SCOPE: this module operates on a sample whose three required learned
stream features have ALREADY been precomputed to
.npy files by the existing per-stream pipelines (visual/src/,
audio/src/, semantic/src/, visual/src/blink_lipsync_precompute.py) -
exactly like every other inference/evaluation script in this repo
(fusion/fusion_inference.py). It
does NOT run raw feature extraction on an arbitrary new video file:
doing that end-to-end ("upload any video, get a verdict") requires
invoking each stream's raw extraction step first (visual frame
sampling + EfficientNet, audio Wav2Vec2, Whisper + Sentence-BERT, and
running the blink/lip-sync analyzers directly on the video). That is
the separate "end-to-end pipeline.py" phase in the project roadmap and
is intentionally NOT built here - see the correction report for why.

Because of that scope, frame-level and windowed lip-sync evidence
(which need the raw video, not just the precomputed 2-d/4-d feature
vectors) are OPTIONAL here: pass video_path (+ frame_output_dir) to
also get frame evidence. When video_path is supplied, this module
itself calls the existing blink_analyzer (compute_ear_series() +
detect_blinks(), the same functions analyze_blinks() itself calls -
not a reimplementation) to obtain genuine blink-event timing for
eye_blink_event-tagged frames, unless you already have blink_events
and pass them in explicitly. If blink detection fails or finds no
events, frame selection falls back to the existing uniform_sample
behavior - see select_evidence_frame_indices() in frame_evidence.py.
Windowed lip-sync evidence is computed the same way: when video_path
is supplied, this module calls the existing
lipsync_analyzer.analyze_lipsync(video_path, compute_windows=True)
directly and returns its genuine window_evidence - it does NOT
fabricate timestamps from the precomputed 2-d vector, and says so
explicitly in its output rather than silently omitting the key when
no video is available.

MISSING CHECKPOINTS: visual/audio/semantic classifier weights and the
three-stream fusion checkpoint must be trained and supplied. require_file() turns a missing checkpoint into a
clear, actionable RuntimeError up front (naming exactly which file is
missing) instead of an obscure torch.load/FileNotFoundError deep in
the pipeline. This module never silently substitutes random weights
and never falls back to a checkpoint with a different architecture.
"""

import json
import logging
import hashlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("fusion", "classifiers", "evidence", "visual/src/lipsync", "visual/src/eye_blink"):
    _p = str(REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from single_stream_classifier import SingleStreamClassifier  # noqa: E402
from fusion_model import FusionModel  # noqa: E402
from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_CLASSIFIER_WEIGHTS,
    DEFAULT_AUDIO_ROOT,
    DEFAULT_BLINK_ROOT,
    DEFAULT_FUSION_WEIGHTS,
    DEFAULT_LIPSYNC_ROOT,
    DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS,
    DEFAULT_SEMANTIC_ROOT,
    DEFAULT_VISUAL_CLASSIFIER_WEIGHTS,
    DEFAULT_VISUAL_ROOT,
)
from evaluate_blink_lipsync import BLINK_ANOMALY_THRESHOLD  # noqa: E402
from lipsync_analyzer import analyze_lipsync, lipsync_status_from_consistency  # noqa: E402
from blink_analyzer import compute_ear_series, detect_blinks  # noqa: E402

from evidence_builder import build_evidence_report  # noqa: E402
from frame_evidence import extract_evidence_frames  # noqa: E402

WINDOW_EVIDENCE_UNAVAILABLE_NOTE = (
    "NOT AVAILABLE: window evidence requires the original video (re-run "
    "lipsync_analyzer.analyze_lipsync(video_path, compute_windows=True)) - it "
    "cannot be recovered from the precomputed 2-d lipsync feature vector alone."
)

DEFAULT_THRESHOLDS_PATH = REPO_ROOT / "eval" / "results" / "thresholds.json"
DEFAULT_DECISION_THRESHOLD = 0.5
REAL_CLASS_INDEX = 0
FAKE_CLASS_INDEX = 1
logger = logging.getLogger(__name__)


def binary_class_probabilities(logits):
    """Convert two-class logits to (P(real), P(fake)) in the training label order."""
    if not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.shape[1] != 2:
        shape = tuple(logits.shape) if isinstance(logits, torch.Tensor) else type(logits).__name__
        raise ValueError(f"Expected binary classifier logits with shape (batch, 2), got {shape}.")
    if not torch.isfinite(logits).all():
        raise ValueError("Classifier logits contain NaN or infinite values.")
    probabilities = torch.softmax(logits, dim=1)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Classifier softmax produced NaN or infinite probabilities.")
    return probabilities


class MissingCheckpointError(FileNotFoundError):
    """Raised by require_file() for a missing trained-model checkpoint -
    distinct from an arbitrary FileNotFoundError so callers can catch
    specifically "training hasn't produced this yet" if they want to."""


class IncompatibleCheckpointError(RuntimeError):
    """A present fusion checkpoint cannot be verified as the active model."""


def require_file(path, description):
    """
    Fail clearly and immediately if a required file (a checkpoint, most
    often) is missing, instead of letting torch.load() raise an obscure
    FileNotFoundError later, deep inside a model-loading call. Returns
    the resolved Path on success so it can be used inline.
    """
    p = Path(path)
    if not p.exists():
        raise MissingCheckpointError(
            f"Missing {description}: {p}\n"
            "This checkpoint has not been produced yet - it is created by running the corresponding "
            "training script (see the README's training protocol) on real data, not by this module."
        )
    return p


def threshold_key_for_checkpoint(weights_path):
    return "fusion_3_stream" if Path(weights_path).name == "best_fusion_model.pt" else None


def load_calibrated_threshold(weights_path, thresholds_path=DEFAULT_THRESHOLDS_PATH):
    """
    Load the validation-calibrated decision threshold for the exact
    three-stream checkpoint being used. Falls back to 0.5 only when
    the threshold artifact or matching checkpoint entry is unavailable.
    """
    threshold_source = {
        "path": str(thresholds_path),
        "key": None,
        "criterion": None,
        "status": "default_0.5",
    }
    thresholds_path = Path(thresholds_path)
    key = threshold_key_for_checkpoint(weights_path)
    threshold_source["key"] = key
    if key is None or not thresholds_path.exists():
        return DEFAULT_DECISION_THRESHOLD, threshold_source
    with thresholds_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)
    entry = doc.get("thresholds", {}).get(key)
    if entry is None:
        return DEFAULT_DECISION_THRESHOLD, threshold_source
    if Path(entry.get("weights", "")).name != Path(weights_path).name:
        return DEFAULT_DECISION_THRESHOLD, threshold_source
    if entry.get("checkpoint_sha256"):
        digest = hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()
        if digest.casefold() != str(entry["checkpoint_sha256"]).casefold():
            return DEFAULT_DECISION_THRESHOLD, threshold_source
    threshold_source.update({
        "criterion": doc.get("criterion"),
        "status": "loaded",
        "validation_balanced_accuracy": entry.get("validation_balanced_accuracy"),
        "validation_f1": entry.get("validation_f1"),
    })
    return float(entry["threshold"]), threshold_source


def _blink_status(irregularity_score, threshold=BLINK_ANOMALY_THRESHOLD):
    """Same boundary rule blink_analyzer.py's analyze_blinks() and
    fusion/evaluate_blink_lipsync.py use (">="), applied here to a
    precomputed vector rather than a live analysis."""
    return "Abnormal" if irregularity_score >= threshold else "Normal"


EXPECTED_FEATURE_SHAPES = {
    "visual": (1280,),
    "audio": (768,),
    "semantic": (384,),
    "blink": (4,),
    "lipsync": (2,),
}


class MissingFeatureFileError(FileNotFoundError):
    """Raised when a required learned-stream feature is absent."""


class FeatureShapeError(ValueError):
    """Raised when a loaded feature vector's shape does not match what
    the classifiers/fusion model were built and trained for."""


def require_feature_file(path, modality_name):
    """Fail clearly when a required learned-stream feature is missing."""
    p = Path(path)
    if not p.exists():
        raise MissingFeatureFileError(
            f"Missing {modality_name} feature file for this sample: {p}\n"
            "Visual, audio and semantic features are required for three-stream inference."
        )
    return p


def load_feature_vector(path, modality_name):
    """
    Loads and validates one modality's precomputed feature vector:
    confirms the file exists (require_feature_file) and that its shape
    matches EXPECTED_FEATURE_SHAPES, so a missing or malformed sample
    fails here with a clear message naming the modality and the exact
    path checked - not deep inside a Linear layer with a cryptic matmul
    shape error, and not silently.
    """
    require_feature_file(path, modality_name)
    vector = np.load(path).astype(np.float32)
    expected_shape = EXPECTED_FEATURE_SHAPES[modality_name]
    if vector.shape != expected_shape:
        raise FeatureShapeError(
            f"{modality_name} feature at {path} has shape {vector.shape}, expected {expected_shape}."
        )
    if not np.isfinite(vector).all():
        raise FeatureShapeError(f"{modality_name} feature at {path} contains NaN or infinite values.")
    return vector


def feature_statistics(feature):
    """Return compact, JSON-safe diagnostics without changing a feature."""
    return {
        "shape": list(feature.shape),
        "dtype": str(feature.dtype),
        "min": float(np.min(feature)),
        "max": float(np.max(feature)),
        "mean": float(np.mean(feature)),
        "std": float(np.std(feature)),
        "finite": bool(np.isfinite(feature).all()),
        "zero_fraction": float(np.mean(feature == 0)),
    }


def load_single_stream_probability(feature, weights_path, input_dim, return_logits=False):
    """
    feature: an already-loaded, already-shape-validated np.ndarray for
    this stream (see load_feature_vector) - loaded once by the caller
    and reused here, rather than this function loading the same .npy
    file a second time.
    """
    model = SingleStreamClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    with torch.inference_mode():
        logits = model(torch.from_numpy(feature).unsqueeze(0))
        probabilities = binary_class_probabilities(logits)
        prob = probabilities[0, FAKE_CLASS_INDEX].item()
    if return_logits:
        return prob, [float(value) for value in logits[0].tolist()]
    return prob


def load_fusion_model(weights_path):
    """Load only a checksum-verified three-stream checkpoint, strictly."""
    model = FusionModel()
    metadata_path = Path(str(weights_path) + ".model_meta.json")
    if not metadata_path.is_file():
        raise IncompatibleCheckpointError(f"Fusion checkpoint metadata is required: {metadata_path}")
    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise IncompatibleCheckpointError("Fusion checkpoint metadata is missing or invalid JSON.") from exc
    if metadata.get("model_class") != "FusionModel" or metadata.get("modalities") != ["visual", "audio", "semantic"]:
        raise IncompatibleCheckpointError("Fusion checkpoint metadata is not for the required three-stream model.")
    digest = hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()
    if str(metadata.get("checkpoint_sha256", "")).casefold() != digest.casefold():
        raise IncompatibleCheckpointError("Fusion checkpoint does not match its metadata checksum.")
    state_dict = torch.load(weights_path, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise IncompatibleCheckpointError("Fusion checkpoint parameters do not match FusionModel.") from exc
    model.eval()
    return model


def obtain_blink_events(video_path):
    """
    Reuse blink_analyzer.py's own functions (compute_ear_series() +
    detect_blinks() - the same two calls analyze_blinks() itself makes)
    to get genuine blink-event timing for a video, without duplicating
    the EAR-based blink-detection algorithm here.

    Returns a list[tuple[int, int]] of (start_frame, end_frame) blink
    events, or an empty list if detection fails or finds nothing - the
    caller (run_full_inference) treats an empty list the same as "no
    blink_events supplied", i.e. frame_evidence.py's existing
    uniform_sample fallback, never an invented event.
    """
    try:
        ear_data = compute_ear_series(video_path)
        return detect_blinks(ear_data["ear_values"])
    except Exception:  # noqa: BLE001 - genuinely optional enrichment, never fatal
        return []


def run_full_inference(
    relative_path,
    visual_root=DEFAULT_VISUAL_ROOT,
    audio_root=DEFAULT_AUDIO_ROOT,
    semantic_root=DEFAULT_SEMANTIC_ROOT,
    blink_root=DEFAULT_BLINK_ROOT,
    lipsync_root=DEFAULT_LIPSYNC_ROOT,
    visual_classifier_weights=DEFAULT_VISUAL_CLASSIFIER_WEIGHTS,
    audio_classifier_weights=DEFAULT_AUDIO_CLASSIFIER_WEIGHTS,
    semantic_classifier_weights=DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS,
    fusion_weights=DEFAULT_FUSION_WEIGHTS,
    thresholds_path=DEFAULT_THRESHOLDS_PATH,
    video_path=None,
    frame_output_dir=None,
    blink_events=None,
    precomputed_lipsync_result=None,
):
    """
    relative_path: path relative to each of the five feature roots,
        e.g. "FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.npy".

    video_path / frame_output_dir / blink_events: all OPTIONAL.
        Omitting video_path skips frame evidence and windowed lip-sync
        evidence entirely - everything else in the result is
        unaffected, and window_evidence/evidence_availability say so
        explicitly rather than silently omitting the keys. Passing
        video_path (+ frame_output_dir) additionally: extracts evidence
        frames (see evidence/frame_evidence.py), using real blink-event
        timing obtained automatically via obtain_blink_events() unless
        you already have events and pass blink_events yourself; and
        computes genuine windowed lip-sync evidence via
        lipsync_analyzer.analyze_lipsync(video_path, compute_windows=True).

    Returns a dict - see module docstring for the full schema and the
    scope note about what requires the raw video vs. what works from
    precomputed features alone.
    """
    rel = Path(relative_path)

    # Fail clearly, up front, if any required checkpoint is missing -
    # rather than a confusing torch.load error later. Never silently
    # substitute random weights or the old 3-modal checkpoint.
    require_file(visual_classifier_weights, "visual classifier checkpoint")
    require_file(audio_classifier_weights, "audio classifier checkpoint")
    require_file(semantic_classifier_weights, "semantic classifier checkpoint")
    require_file(fusion_weights, "three-stream fusion model checkpoint")

    # Validate every modality's feature file for this sample exists and
    # has the expected shape BEFORE anything is loaded into a
    # classifier or the fusion model - an incomplete or malformed
    # sample must never reach either silently.
    visual_feat = load_feature_vector(Path(visual_root) / rel, "visual")
    audio_feat = load_feature_vector(Path(audio_root) / rel, "audio")
    semantic_feat = load_feature_vector(Path(semantic_root) / rel, "semantic")
    blink_vector = None
    lipsync_vector = None
    try:
        blink_vector = load_feature_vector(Path(blink_root) / rel, "blink")
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Supplementary blink result unavailable for %s: %s", rel.name, exc)
    try:
        lipsync_vector = load_feature_vector(Path(lipsync_root) / rel, "lipsync")
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Supplementary lip-sync result unavailable for %s: %s", rel.name, exc)

    classifier_started = time.monotonic()
    visual_fake_probability, visual_logits = load_single_stream_probability(
        visual_feat, visual_classifier_weights, 1280, return_logits=True
    )
    audio_fake_probability, audio_logits = load_single_stream_probability(
        audio_feat, audio_classifier_weights, 768, return_logits=True
    )
    semantic_fake_probability, semantic_logits = load_single_stream_probability(
        semantic_feat, semantic_classifier_weights, 384, return_logits=True
    )
    logger.info(
        "Loaded three learned-stream checkpoints: visual=%s audio=%s semantic=%s",
        Path(visual_classifier_weights).name,
        Path(audio_classifier_weights).name,
        Path(semantic_classifier_weights).name,
    )
    logger.info(
        "Validated learned streams for sample=%s: visual=%s audio=%s semantic=%s",
        rel.name,
        visual_feat.shape,
        audio_feat.shape,
        semantic_feat.shape,
    )
    classifier_inference_seconds = time.monotonic() - classifier_started

    # blink vector: [blink_count, blink_rate_per_min, avg_blink_duration_sec, blink_irregularity_score]
    tensors = [
        torch.from_numpy(x).unsqueeze(0)
        for x in (visual_feat, audio_feat, semantic_feat)
    ]

    fusion_started = time.monotonic()
    fusion_model = load_fusion_model(fusion_weights)

    with torch.inference_mode():
        logits, attention = fusion_model(*tensors)
        probs = binary_class_probabilities(logits)
        final_fake_probability = round(probs[0, FAKE_CLASS_INDEX].item(), 6)
        final_real_probability = round(probs[0, REAL_CLASS_INDEX].item(), 6)
    fusion_inference_seconds = time.monotonic() - fusion_started
    decision_threshold, threshold_source = load_calibrated_threshold(fusion_weights, thresholds_path)
    prediction = "DEEPFAKE" if final_fake_probability >= decision_threshold else "REAL"

    logger.info(
        "Inference scores (%s): visual logits=%s fake=%.6f; audio logits=%s fake=%.6f; "
        "semantic logits=%s fake=%.6f; fusion checkpoint=%s architecture=%s logits=%s "
        "P(real)=%.6f P(fake)=%.6f threshold=%.6f (%s) prediction=%s",
        rel.name,
        [round(value, 6) for value in visual_logits], visual_fake_probability,
        [round(value, 6) for value in audio_logits], audio_fake_probability,
        [round(value, 6) for value in semantic_logits], semantic_fake_probability,
        Path(fusion_weights).name,
        "three_stream",
        [round(float(value), 6) for value in logits[0].cpu().tolist()],
        final_real_probability, final_fake_probability,
        decision_threshold, threshold_source.get("status"), prediction,
    )

    evidence_started = time.monotonic()

    blink_irregularity_score = float(blink_vector[3]) if blink_vector is not None else None
    blink_status = _blink_status(blink_irregularity_score) if blink_irregularity_score is not None else "Unavailable"
    lipsync_consistency = float(lipsync_vector[0]) if lipsync_vector is not None else None
    lip_sync_mismatch_score = float(lipsync_vector[1]) if lipsync_vector is not None else None
    lip_sync_status = lipsync_status_from_consistency(lipsync_consistency) if lipsync_consistency is not None else "Unavailable"

    evidence = build_evidence_report(
        visual_fake_probability=visual_fake_probability,
        audio_fake_probability=audio_fake_probability,
        semantic_fake_probability=semantic_fake_probability,
        blink_result={"blink_irregularity_score": blink_irregularity_score, "blink_status": blink_status},
        lipsync_result={"mismatch_score": lip_sync_mismatch_score, "lipsync_status": lip_sync_status},
        final_fake_probability=final_fake_probability,
        decision_threshold=decision_threshold,
    )

    # Blink-event frames: if a raw video was supplied and the caller
    # didn't already hand us blink_events, obtain genuine ones ourselves
    # by reusing blink_analyzer.py's own functions (see
    # obtain_blink_events()'s docstring) - never invented, and never a
    # reimplementation of the EAR-based detection algorithm. An empty
    # result (detection failed or found nothing) falls back to
    # frame_evidence.py's existing uniform_sample behavior, unchanged.
    used_real_blink_events = False
    if video_path is not None and blink_events is None:
        blink_events = obtain_blink_events(video_path)
        used_real_blink_events = bool(blink_events)

    frame_evidence = None
    if video_path is not None and frame_output_dir is not None:
        try:
            frame_evidence = extract_evidence_frames(video_path, frame_output_dir, blink_events=blink_events)
            if blink_events:
                used_real_blink_events = True
        except Exception as exc:  # evidence failure must not invalidate fusion scores
            logger.warning("Visual context evidence unavailable: %s", exc)

    # Windowed lip-sync evidence: only genuinely computable with the raw
    # video (analyze_lipsync's correlation needs the actual mouth/audio
    # time series, not the single precomputed 2-d summary vector). When
    # video_path is supplied, call the existing analyzer directly and
    # return its real window_evidence; otherwise say plainly that it is
    # unavailable rather than fabricating timestamps. The precomputed
    # lip_sync_mismatch_score above is left as the reported global
    # score in both cases - this does not recompute it differently.
    window_evidence = WINDOW_EVIDENCE_UNAVAILABLE_NOTE
    lipsync_window_evidence_available = False
    if precomputed_lipsync_result is not None:
        window_evidence = precomputed_lipsync_result.get("window_evidence", [])
        lipsync_window_evidence_available = True
    elif video_path is not None:
        try:
            lipsync_result = analyze_lipsync(video_path, compute_windows=True)
            window_evidence = lipsync_result.get("window_evidence", [])
            lipsync_window_evidence_available = True
        except Exception as exc:  # noqa: BLE001 - report, don't fabricate, don't crash the whole call
            window_evidence = f"NOT AVAILABLE: window evidence computation failed on the supplied video ({exc})."

    evidence_availability = {
        "numeric_stream_scores": True,
        "frame_evidence": frame_evidence is not None,
        "blink_event_frames": used_real_blink_events,
        "lip_sync_window_evidence": lipsync_window_evidence_available,
        "visual_artifact_detection": False,
    }
    evidence_generation_seconds = time.monotonic() - evidence_started

    return {
        "final_fake_probability": final_fake_probability,
        "final_real_probability": final_real_probability,
        "prediction": prediction,
        "visual_fake_probability": round(visual_fake_probability, 4),
        "audio_fake_probability": round(audio_fake_probability, 4),
        "semantic_fake_probability": round(semantic_fake_probability, 4),
        "blink_anomaly_score": round(blink_irregularity_score, 4) if blink_irregularity_score is not None else None,
        "blink_status": blink_status,
        "lip_sync_mismatch_score": round(lip_sync_mismatch_score, 4) if lip_sync_mismatch_score is not None else None,
        "lip_sync_status": lip_sync_status,
        "supplementary_consistency": {
            "blink": {"score": round(blink_irregularity_score, 4) if blink_irregularity_score is not None else None, "status": blink_status},
            "lip_sync": {"mismatch_score": round(lip_sync_mismatch_score, 4) if lip_sync_mismatch_score is not None else None, "status": lip_sync_status},
            "contributes_to_final_result": False,
        },
        "individual_stream_probabilities": {
            "visual": round(visual_fake_probability, 4),
            "audio": round(audio_fake_probability, 4),
            "semantic": round(semantic_fake_probability, 4),
        },
        "decision_threshold": round(float(decision_threshold), 6),
        "evidence": evidence,
        "frame_evidence": frame_evidence,
        "window_evidence": window_evidence,
        "evidence_availability": evidence_availability,
        "_inference_timing_seconds": {
            "classifier_inference": classifier_inference_seconds,
            "fusion_inference": fusion_inference_seconds,
            "evidence_generation": evidence_generation_seconds,
        },
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run three-stream inference on one precomputed sample."
    )
    parser.add_argument("--relative-path", required=True)
    parser.add_argument("--visual-root", default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--semantic-root", default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--blink-root", default=DEFAULT_BLINK_ROOT)
    parser.add_argument("--lipsync-root", default=DEFAULT_LIPSYNC_ROOT)
    parser.add_argument("--visual-classifier-weights", default=DEFAULT_VISUAL_CLASSIFIER_WEIGHTS)
    parser.add_argument("--audio-classifier-weights", default=DEFAULT_AUDIO_CLASSIFIER_WEIGHTS)
    parser.add_argument("--semantic-classifier-weights", default=DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS)
    parser.add_argument("--fusion-weights", default=DEFAULT_FUSION_WEIGHTS)
    parser.add_argument("--thresholds-path", default=str(DEFAULT_THRESHOLDS_PATH))
    args = parser.parse_args()

    result = run_full_inference(
        args.relative_path,
        visual_root=args.visual_root, audio_root=args.audio_root, semantic_root=args.semantic_root,
        blink_root=args.blink_root, lipsync_root=args.lipsync_root,
        visual_classifier_weights=args.visual_classifier_weights,
        audio_classifier_weights=args.audio_classifier_weights,
        semantic_classifier_weights=args.semantic_classifier_weights,
        fusion_weights=args.fusion_weights,
        thresholds_path=args.thresholds_path,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
