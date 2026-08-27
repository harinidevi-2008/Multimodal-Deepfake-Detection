"""
run_full_inference() result -> frontend API_CONTRACT.md shape.

The ONE place field-name conversions happen (per the integration spec's
own instruction: "write one adapter/serializer... don't scatter
field-name conversions across React components"). Every value here is
either read straight from the pipeline's real result dict, or a static
explanatory string (attention/contribution "note" fields, the
low-confidence-margin note) - nothing here is a fabricated number.

Naming note: run_full_inference()'s result has a top-level "evidence"
key that is evidence/evidence_builder.py's plain-language report dict
(verdict, low_confidence, a list of reason strings, etc) - a totally
different thing from the frontend contract's top-level "evidence" key
(frames/audio_waveform/lip_sync_window_evidence/blink_timeline). To
keep that straight, this module calls the former `evidence_report` and
reserves `evidence` for the contract's evidence block.
"""

from pathlib import Path

MODALITY_LABELS = {
    "visual": "Visual",
    "audio": "Audio",
    "semantic": "Semantic",
}

ATTENTION_NOTE = (
    "Descriptive cross-attention weights (mean incoming attention per modality "
    "token) - not a causal measure of importance. See fusion/attention_utils.py."
)
CONTRIBUTION_NOTE = (
    "Ablation-based: each value is the drop in the final fake probability when "
    "that modality's input is zeroed out. A heuristic sensitivity probe, not a "
    "ground-truth attribution. See fusion/modality_contribution.py."
)
MODEL_CONFIDENCE_NOTE = (
    "Model confidence reflects distance from the 0.5 decision boundary - a final "
    "fake probability within 0.15 of 0.5 is flagged as low-confidence (see "
    "evidence/evidence_builder.py's LOW_CONFIDENCE_MARGIN)."
)


def _rename_modality_keys(d):
    """attention/contribution dicts key modalities as "blink"/"lipsync"
    (MODALITY_ORDER_5) - the frontend contract (and ModalityCard.jsx /
    modalityMeta.js) uses "eye_blink"/"lip_sync" everywhere else. Renamed
    only here, once, rather than in every component."""
    out = dict(d)
    if "blink" in out:
        out["eye_blink"] = out.pop("blink")
    if "lipsync" in out:
        out["lip_sync"] = out.pop("lipsync")
    return out


def _build_frames(frame_evidence, fps, job_id):
    if not frame_evidence:
        return None
    frames = []
    for item in frame_evidence:
        if not item.get("saved_path"):
            # Frame could not be read/saved (see frame_evidence.py) -
            # nothing real to serve, so it's dropped rather than given
            # a broken or fabricated url.
            continue
        frames.append({
            "frame_index": item["frame_index"],
            "timestamp_seconds": round(item["frame_index"] / fps, 3) if fps else None,
            "url": f"/api/evidence/{job_id}/{Path(item['saved_path']).name}",
            "frame_selection_reason": item["frame_selection_reason"],
            "detected_artifact": item.get("detected_artifact"),  # always None until a real detector exists
        })
    return frames or None


def _build_lip_sync_window_evidence(window_evidence):
    # window_evidence is either a list of real windows (analyze_lipsync
    # succeeded with compute_windows=True) or a string explaining why it
    # is unavailable (see full_pipeline.WINDOW_EVIDENCE_UNAVAILABLE_NOTE
    # or the per-call failure message) - only the list case is genuine,
    # renderable evidence; the string case means "omit the key", per
    # API_CONTRACT.md, not "fabricate an empty chart".
    if not isinstance(window_evidence, list) or not window_evidence:
        return None
    return [
        {
            "window_start_seconds": w.get("start_sec"),
            "window_end_seconds": w.get("end_sec"),
            "mismatch_score": w.get("mismatch_score"),
        }
        for w in window_evidence
    ]


def _build_contextual_analysis(evidence_report, lip_sync_status, blink_status):
    reasons = evidence_report.get("evidence", [])
    detected_mismatch = None
    if lip_sync_status == "Inconsistent":
        detected_mismatch = (
            "Audio-visual synchronization inconsistency detected (rule-based lip-sync mismatch score)."
        )
    elif blink_status == "Abnormal":
        detected_mismatch = "Abnormal eye-blink pattern detected (rule-based anomaly score)."
    return {
        "detected_mismatch": detected_mismatch,
        "explanation": " ".join(reasons) if reasons else None,
    }


def serialize_result(result, meta, job_id, video_filename, processing_time_seconds, analyzed_at):
    """
    result: run_full_inference()'s native dict, plus the adapter's own
        "_blink_timeline" key (see pipeline_adapter.run_raw_video_analysis).
    meta: {"fps", "total_frames", "duration_seconds"} from
        pipeline_adapter._probe_video().
    job_id: this request's job id (for building evidence URLs).
    video_filename: the ORIGINAL uploaded filename (server.py has this
        from the multipart field - not derivable from the job's saved
        path, which is deliberately not filename-based).
    processing_time_seconds: wall-clock time server.py measured around
        the pipeline call.
    analyzed_at: ISO-8601 timestamp string server.py generated at the
        time the request was handled.
    """
    evidence_report = result["evidence"]
    fps = meta.get("fps") or 0.0

    modalities = {
        "visual": {
            "label": MODALITY_LABELS["visual"],
            "kind": "learned_probability",
            "fake_probability": result["visual_fake_probability"],
        },
        "audio": {
            "label": MODALITY_LABELS["audio"],
            "kind": "learned_probability",
            "fake_probability": result["audio_fake_probability"],
        },
        "semantic": {
            "label": MODALITY_LABELS["semantic"],
            "kind": "learned_probability",
            "fake_probability": result["semantic_fake_probability"],
        },
        "eye_blink": {
            "label": "Eye Blink",
            "kind": "rule_based_score",
            "anomaly_score": result["blink_anomaly_score"],
            "status": result["blink_status"],
        },
        "lip_sync": {
            "label": "Lip Sync",
            "kind": "rule_based_score",
            "mismatch_score": result["lip_sync_mismatch_score"],
            "status": result["lip_sync_status"],
        },
    }

    evidence = {}
    frames = _build_frames(result.get("frame_evidence"), fps, job_id)
    if frames:
        evidence["frames"] = frames
    # No genuine audio waveform envelope is computed anywhere in this
    # pipeline (audio/src/audio_stream.py only ever produces the pooled
    # 768-d Wav2Vec2 feature, never a sample-level amplitude envelope) -
    # omit the key entirely rather than fabricate one. See API_CONTRACT.md's
    # "omit a key rather than fabricating" instruction.
    window_evidence = _build_lip_sync_window_evidence(result.get("window_evidence"))
    if window_evidence:
        evidence["lip_sync_window_evidence"] = window_evidence
    blink_timeline = result.get("_blink_timeline")
    if blink_timeline:
        evidence["blink_timeline"] = blink_timeline

    return {
        "video_filename": video_filename,
        "video_duration_seconds": meta.get("duration_seconds"),
        "analyzed_at": analyzed_at,
        "processing_time_seconds": round(processing_time_seconds, 3),
        "final_verdict": "LIKELY_DEEPFAKE" if result["prediction"] == "DEEPFAKE" else "LIKELY_REAL",
        "final_fake_probability": result["final_fake_probability"],
        "final_real_probability": result["final_real_probability"],
        "low_confidence": evidence_report.get("low_confidence"),
        "model_confidence_note": MODEL_CONFIDENCE_NOTE,
        "modalities": modalities,
        "attention_summary": {
            "weights": _rename_modality_keys(result["attention_summary"]),
            "note": ATTENTION_NOTE,
        },
        "modality_contributions": {
            "deltas": _rename_modality_keys(result["modality_contributions"].get("contributions", {})),
            "note": CONTRIBUTION_NOTE,
        },
        "evidence": evidence,
        "contextual_analysis": _build_contextual_analysis(
            evidence_report, result["lip_sync_status"], result["blink_status"]
        ),
    }
