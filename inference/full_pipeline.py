"""
Application-facing inference orchestration layer.

This does NOT reimplement anything - it wires together modules that
already exist and already work on their own:
  - classifiers/single_stream_classifier.py  (visual/audio/semantic P(fake), each its own trained model)
  - fusion/enhanced_fusion_model.py          (final 5-modal P(fake))
  - fusion/feature_normalization.py          (the SAME blink/lipsync normalization used in training)
  - fusion/attention_utils.py                (descriptive, non-causal attention summary)
  - fusion/modality_contribution.py          (ablation-based modality contribution - a heuristic, not proof)
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

SCOPE (read before wiring this into a UI): this module operates on a
sample whose five modality features have ALREADY been precomputed to
.npy files by the existing per-stream pipelines (visual/src/,
audio/src/, semantic/src/, visual/src/blink_lipsync_precompute.py) -
exactly like every other inference/evaluation script in this repo
(fusion/fusion_inference.py, fusion/enhanced_fusion_inference.py). It
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
enhanced fusion model weights will not exist until real training has
been run. require_file() below turns a missing checkpoint into a
clear, actionable RuntimeError up front (naming exactly which file is
missing) instead of an obscure torch.load/FileNotFoundError deep in
the pipeline. This module never silently substitutes random weights
or falls back to the old 3-modal checkpoint for a missing 5-modal one.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _sub in ("fusion", "classifiers", "evidence", "visual/src/lipsync", "visual/src/eye_blink"):
    _p = str(REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from single_stream_classifier import SingleStreamClassifier  # noqa: E402
from enhanced_fusion_model import EnhancedFusionModel  # noqa: E402
from feature_normalization import (  # noqa: E402
    DEFAULT_NORMALIZATION_PATH,
    apply_normalization,
    check_normalization_consistency,
    load_normalization,
)
from attention_utils import MODALITY_ORDER_5, summarize_attention  # noqa: E402
from modality_contribution import modality_contributions  # noqa: E402
from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_CLASSIFIER_WEIGHTS,
    DEFAULT_AUDIO_ROOT,
    DEFAULT_BLINK_ROOT,
    DEFAULT_ENHANCED_FUSION_WEIGHTS,
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


class MissingCheckpointError(FileNotFoundError):
    """Raised by require_file() for a missing trained-model checkpoint -
    distinct from an arbitrary FileNotFoundError so callers can catch
    specifically "training hasn't produced this yet" if they want to."""


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
    """Raised when a modality's precomputed feature file is missing for
    this specific sample - distinct from MissingCheckpointError, which
    is about trained-model weights, not per-sample feature data."""


class FeatureShapeError(ValueError):
    """Raised when a loaded feature vector's shape does not match what
    the classifiers/fusion model were built and trained for."""


def require_feature_file(path, modality_name):
    """
    Fail clearly and immediately if a modality's precomputed feature
    file for this sample is missing, instead of letting np.load() raise
    a bare FileNotFoundError with no indication of which of the five
    modalities was the culprit. An incomplete sample must never be
    allowed to silently reach a classifier or the fusion model - this
    is the per-sample-feature analogue of require_file()'s checkpoint
    check above.
    """
    p = Path(path)
    if not p.exists():
        raise MissingFeatureFileError(
            f"Missing {modality_name} feature file for this sample: {p}\n"
            "All five modality feature files (visual/audio/semantic/blink/lipsync) must exist "
            "for this sample before it can be run through any classifier or the fusion model."
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
    return vector


def load_single_stream_probability(feature, weights_path, input_dim):
    """
    feature: an already-loaded, already-shape-validated np.ndarray for
    this stream (see load_feature_vector) - loaded once by the caller
    and reused here, rather than this function loading the same .npy
    file a second time.
    """
    model = SingleStreamClassifier(input_dim=input_dim)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        logits = model(torch.from_numpy(feature).unsqueeze(0))
        prob = torch.softmax(logits, dim=1)[0, 1].item()
    return prob


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
    enhanced_fusion_weights=DEFAULT_ENHANCED_FUSION_WEIGHTS,
    normalization_path=DEFAULT_NORMALIZATION_PATH,
    video_path=None,
    frame_output_dir=None,
    blink_events=None,
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
    require_file(enhanced_fusion_weights, "enhanced (5-modal) fusion model checkpoint")

    # Validate every modality's feature file for this sample exists and
    # has the expected shape BEFORE anything is loaded into a
    # classifier or the fusion model - an incomplete or malformed
    # sample must never reach either silently.
    visual_feat = load_feature_vector(Path(visual_root) / rel, "visual")
    audio_feat = load_feature_vector(Path(audio_root) / rel, "audio")
    semantic_feat = load_feature_vector(Path(semantic_root) / rel, "semantic")
    blink_vector = load_feature_vector(Path(blink_root) / rel, "blink")
    lipsync_vector = load_feature_vector(Path(lipsync_root) / rel, "lipsync")

    visual_fake_probability = load_single_stream_probability(
        visual_feat, visual_classifier_weights, 1280
    )
    audio_fake_probability = load_single_stream_probability(
        audio_feat, audio_classifier_weights, 768
    )
    semantic_fake_probability = load_single_stream_probability(
        semantic_feat, semantic_classifier_weights, 384
    )

    # blink vector: [blink_count, blink_rate_per_min, avg_blink_duration_sec, blink_irregularity_score]
    blink_irregularity_score = float(blink_vector[3])
    blink_status = _blink_status(blink_irregularity_score)

    # lipsync vector: [sync_score (= consistency), mismatch_score]
    lipsync_consistency = float(lipsync_vector[0])
    lip_sync_mismatch_score = float(lipsync_vector[1])
    lip_sync_status = lipsync_status_from_consistency(lipsync_consistency)

    normalization = None
    normalization_used = False
    if normalization_path is not None and Path(normalization_path).exists():
        normalization = load_normalization(normalization_path)
        normalization_used = True

    # Detect a checkpoint trained with (or without) normalization being
    # run here with the opposite setting - see feature_normalization.py's
    # check_normalization_consistency() docstring. A warning, not a hard
    # failure: an unrecognized/pre-existing checkpoint (no sidecar yet)
    # must not block inference, but the mismatch must not be silent.
    normalization_warning = check_normalization_consistency(enhanced_fusion_weights, normalization_used)
    if normalization_warning:
        if normalization_warning.startswith("[normalization MISMATCH]"):
            # A CONFIRMED mismatch (the sidecar explicitly says one thing,
            # this run is about to do the other) - loud enough that it
            # cannot be mistaken for routine log output, matching
            # evaluate_enhanced_fusion.py's banner treatment of the same
            # condition. The result below is not to be trusted as-is.
            banner = "!" * 70
            print(f"\n{banner}\n{normalization_warning}\n{banner}\n")
        else:
            # No sidecar at all (e.g. a pre-existing checkpoint) - cannot
            # be verified automatically, but is not a confirmed mismatch
            # either; a plain single-line warning is enough, per the
            # existing checkpoint-compatibility policy.
            print(f"\nWARNING: {normalization_warning}\n")

    blink_feat = blink_vector.copy()
    lipsync_feat = lipsync_vector.copy()
    if normalization is not None:
        blink_feat = apply_normalization(blink_feat, normalization["blink"])
        lipsync_feat = apply_normalization(lipsync_feat, normalization["lipsync"])

    tensors = [
        torch.from_numpy(x).unsqueeze(0)
        for x in (visual_feat, audio_feat, semantic_feat, blink_feat, lipsync_feat)
    ]

    fusion_model = EnhancedFusionModel()
    fusion_model.load_state_dict(torch.load(enhanced_fusion_weights, map_location="cpu"))
    fusion_model.eval()

    with torch.no_grad():
        logits, attention = fusion_model(*tensors)
        probs = torch.softmax(logits, dim=1)
        final_fake_probability = probs[0, 1].item()
        final_real_probability = probs[0, 0].item()
    prediction = "DEEPFAKE" if final_fake_probability >= 0.5 else "REAL"

    attention_summary = summarize_attention(attention, MODALITY_ORDER_5)
    contribution = modality_contributions(fusion_model, tensors, MODALITY_ORDER_5)

    evidence = build_evidence_report(
        visual_fake_probability=visual_fake_probability,
        audio_fake_probability=audio_fake_probability,
        semantic_fake_probability=semantic_fake_probability,
        blink_result={"blink_irregularity_score": blink_irregularity_score, "blink_status": blink_status},
        lipsync_result={"mismatch_score": lip_sync_mismatch_score, "lipsync_status": lip_sync_status},
        final_fake_probability=final_fake_probability,
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
        frame_evidence = extract_evidence_frames(video_path, frame_output_dir, blink_events=blink_events)
        if blink_events:
            used_real_blink_events = True

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
    if video_path is not None:
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

    return {
        "final_fake_probability": round(final_fake_probability, 4),
        "final_real_probability": round(final_real_probability, 4),
        "prediction": prediction,
        "visual_fake_probability": round(visual_fake_probability, 4),
        "audio_fake_probability": round(audio_fake_probability, 4),
        "semantic_fake_probability": round(semantic_fake_probability, 4),
        "blink_anomaly_score": round(blink_irregularity_score, 4),
        "blink_status": blink_status,
        "lip_sync_mismatch_score": round(lip_sync_mismatch_score, 4),
        "lip_sync_status": lip_sync_status,
        "attention": attention,
        "attention_summary": attention_summary,
        "modality_contributions": contribution,
        "evidence": evidence,
        "frame_evidence": frame_evidence,
        "window_evidence": window_evidence,
        "normalization_applied": normalization_used,
        "normalization_warning": normalization_warning,
        "evidence_availability": evidence_availability,
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run the full application-facing 5-modal inference pipeline on one precomputed sample."
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
    parser.add_argument("--enhanced-fusion-weights", default=DEFAULT_ENHANCED_FUSION_WEIGHTS)
    parser.add_argument("--normalization-path", default=str(DEFAULT_NORMALIZATION_PATH))
    parser.add_argument("--no-normalization", action="store_true")
    args = parser.parse_args()

    result = run_full_inference(
        args.relative_path,
        visual_root=args.visual_root, audio_root=args.audio_root, semantic_root=args.semantic_root,
        blink_root=args.blink_root, lipsync_root=args.lipsync_root,
        visual_classifier_weights=args.visual_classifier_weights,
        audio_classifier_weights=args.audio_classifier_weights,
        semantic_classifier_weights=args.semantic_classifier_weights,
        enhanced_fusion_weights=args.enhanced_fusion_weights,
        normalization_path=(None if args.no_normalization else args.normalization_path),
    )
    printable = {k: v for k, v in result.items() if k != "attention"}
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
