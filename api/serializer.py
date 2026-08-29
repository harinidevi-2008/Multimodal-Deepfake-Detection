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

from api.errors import InferenceFailedError

MODALITY_LABELS = {
    "visual": "Visual",
    "audio": "Audio",
    "semantic": "Semantic",
}
LEARNED_HIGH_SIGNAL_THRESHOLD = 0.6

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
        evidence_type = "supporting_context"
        explanation = "Visual artifact localization is unavailable. This is a supporting visual sample and is not proof of manipulation."
        if item.get("detected_artifact") is not None:
            evidence_type = "localized_visual_artifact"
            explanation = str(item["detected_artifact"])
        elif item.get("frame_selection_reason") == "eye_blink_event":
            evidence_type = "blink_event"
            event = item.get("blink_event")
            if event and fps:
                start = event[0] / fps
                end = event[1] / fps
                explanation = f"Frame {item['frame_index']} is shown because it corresponds to a detected blink event at {start:.1f}-{end:.1f}s. This is supporting context, not proof of manipulation."
            else:
                explanation = f"Frame {item['frame_index']} is shown because it corresponds to a detected blink event. This is supporting context, not proof of manipulation."
        frame_record = {
            "frame_index": item["frame_index"],
            "timestamp_seconds": round(item["frame_index"] / fps, 3) if fps else None,
            "url": f"/api/evidence/{job_id}/{Path(item['saved_path']).name}",
            "frame_selection_reason": item["frame_selection_reason"],
            "detected_artifact": item.get("detected_artifact"),  # always None until a real detector exists
            "evidence_type": evidence_type,
            "evidence_source": item.get("frame_selection_reason"),
            "explanation": explanation,
        }
        if item.get("detected_artifact") is not None:
            frame_record["confidence"] = item.get("confidence")
            frame_record["strength"] = item.get("strength")
        frames.append(frame_record)
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
            "threshold": 0.6,
            "anomalous": (w.get("mismatch_score") or 0) > 0.6,
            "evidence_status": "localized" if (w.get("mismatch_score") or 0) > 0.6 else "not_detected",
        }
        for w in window_evidence
    ]


def _score_strength(score, supports_fake):
    if score is None:
        return "unknown"
    score = float(score)
    if supports_fake:
        if score >= 0.8:
            return "strong"
        if score >= 0.6:
            return "moderate"
        return "weak"
    if score <= 0.2:
        return "strong"
    if score <= 0.4:
        return "moderate"
    return "weak"


def _modality_explanations(result, blink_status, lip_sync_status):
    semantic = result.get("_semantic_metadata") or {}
    semantic_health = result.get("_feature_health", {}).get("semantic", {})

    def make_entry(modality, label, supports, score, what_it_analyzes, signal_meaning, limitation, explanation):
        return {
            "modality": modality,
            "label": label,
            "supports": supports,
            "strength": _score_strength(score, supports == "FAKE"),
            "what_it_analyzes": what_it_analyzes,
            "signal_meaning": signal_meaning,
            "limitation": limitation,
            "explanation": explanation,
        }

    entries = [
        make_entry(
            "visual",
            "Visual",
            "FAKE" if result["visual_fake_probability"] >= 0.6 else "REAL",
            result["visual_fake_probability"],
            "The visual stream uses learned frame-level features from the sampled video frames.",
            "This is a classifier signal about the learned visual representation, not a claim that a specific visual artifact was localized in this frame.",
            "No visual artifact detector is active in this pipeline, so a high score does not identify a specific defect in the video.",
            (
                "Visual analysis supports manipulation because the visual representation resembles patterns associated with manipulated samples in training. "
                "This is a learned classifier signal from the visual embedding, not a claim that the displayed frames contain a detected artifact."
                if result["visual_fake_probability"] >= 0.6
                else "Visual analysis supports real content because the learned visual representation resembles patterns associated with authentic samples in training. "
                "This is a learned classifier signal from the visual embedding, not a claim that the displayed frames contain a detected artifact."
            ),
        ),
        make_entry(
            "audio",
            "Audio",
            "FAKE" if result["audio_fake_probability"] >= 0.6 else "REAL",
            result["audio_fake_probability"],
            "The audio stream analyzes the extracted speech representation from the soundtrack.",
            "This indicates the learned audio representation resembles patterns associated with manipulated speech in training, but it does not pinpoint a specific audible artifact.",
            "The score is global unless timestamped suspicious audio windows are explicitly returned; it is not a localized tampering claim.",
            (
                "Audio analysis strongly supports manipulation because the audio classifier assigns a high fake probability. This indicates that the extracted audio representation resembles patterns associated with manipulated samples in training. This score alone does not identify a specific audible artifact."
                if result["audio_fake_probability"] >= 0.6
                else "Audio analysis supports real content because the audio classifier assigns a low fake probability. This indicates the extracted speech representation resembles patterns associated with authentic samples in training. The score is still a global classifier signal and does not identify a specific audio segment as altered."
            ),
        ),
        make_entry(
            "semantic",
            "Semantic",
            "FAKE" if result["semantic_fake_probability"] >= 0.6 else "REAL",
            result["semantic_fake_probability"],
            "The semantic stream analyzes the extracted speech/text embedding and transcript context.",
            "This is a classifier signal about the learned speech representation, not a claim that the spoken words or meaning are fabricated.",
            "Semantic evidence is global unless localized transcript-level anomalies are explicitly returned; transcript quality and confidence also matter.",
            (
                "Semantic analysis supports manipulation because the speech representation produces a high fake score. This is a classifier signal about the learned speech representation, not a claim that the spoken words or meaning are fabricated."
                if result["semantic_fake_probability"] >= 0.6
                else "Semantic analysis supports real content because the speech representation produces a low fake score. This is a classifier signal about the learned speech representation, not a claim that the spoken words or meaning are authentic beyond the model's training distribution."
            )
            + (
                " The transcript is not treated as a localized anomaly unless the backend explicitly provides transcript-level anomaly evidence."
                if not semantic.get("segments")
                else " Transcript segments are presented only as context and are not evidence of semantic manipulation unless an anomaly detector exists."
            )
            if semantic_health.get("status") != "zero_embedding"
            else "Semantic analysis was degraded because the generated embedding was invalid. This makes the signal unreliable, so it should not be interpreted as a semantic anomaly or as a transcript-level manipulation claim.",
        ),
        make_entry(
            "eye_blink",
            "Eye Blink",
            "FAKE" if blink_status == "Abnormal" else "REAL",
            result["blink_anomaly_score"],
            "Eye-blink analysis uses eye-aspect-ratio timing and blink-event statistics across the video.",
            "The rule-based detector is flagging an abnormal blink pattern, which is a behavioral consistency signal rather than proof of manipulation by itself.",
            "Blink anomalies are supporting context only; they do not identify a specific altered frame or guarantee a deepfake.",
            (
                "Blink analysis supports manipulation because the measured blink pattern crossed the configured abnormality criteria. This is a behavioral consistency signal, not a direct detection of a face swap or manipulated frame."
                if blink_status == "Abnormal"
                else "Blink analysis supports real content because the measured blink pattern did not cross the configured abnormality criteria. This is a behavioral consistency signal, not proof that the video is authentic."
            ),
        ),
        make_entry(
            "lip_sync",
            "Lip Sync",
            "FAKE" if lip_sync_status == "Inconsistent" else "REAL",
            result["lip_sync_mismatch_score"],
            "Lip-sync analysis compares mouth motion against the audio track over time windows.",
            "The mismatch score reflects timing consistency between speech and mouth motion, which is a rule-based signal rather than a trained probability.",
            "Windowed mismatch evidence is only valid when the backend has localized lip-sync windows; otherwise the score is an aggregate signal, not a localized artifact claim.",
            (
                "Lip-sync analysis supports manipulation because the overall synchronization pattern is inconsistent with the current rule-based criteria. This is a timing-consistency signal, not a confirmed manipulation artifact by itself."
                if lip_sync_status == "Inconsistent"
                else "Lip-sync analysis supports real content because the overall synchronization pattern remains consistent under the current rule-based criteria. This is a timing-consistency signal, not a confirmed manipulation artifact by itself."
            ),
        ),
    ]
    return entries


def _build_contextual_analysis(result, evidence_report, lip_sync_status, blink_status):
    scores = [
        ("visual", "Visual", result["visual_fake_probability"]),
        ("audio", "Audio", result["audio_fake_probability"]),
        ("semantic", "Semantic", result["semantic_fake_probability"]),
    ]
    fake_support = [
        {"modality": key, "label": name, "score": score, "reason": f"{name} produced a {score:.0%} fake probability."}
        for key, name, score in scores if score >= LEARNED_HIGH_SIGNAL_THRESHOLD
    ]
    real_support = [
        {"modality": key, "label": name, "score": score, "reason": f"{name} produced a {1 - score:.0%} real probability."}
        for key, name, score in scores if score < 0.5
    ]
    if blink_status != "Abnormal":
        real_support.append({
            "modality": "eye_blink",
            "label": "Eye Blink",
            "score": result["blink_anomaly_score"],
            "reason": "Blink analysis did not identify an abnormal pattern.",
        })
    else:
        fake_support.append({
            "modality": "eye_blink",
            "label": "Eye Blink",
            "score": result["blink_anomaly_score"],
            "reason": "Blink analysis classified the aggregate blink pattern as abnormal.",
        })
    if lip_sync_status != "Inconsistent":
        real_support.append({
            "modality": "lip_sync",
            "label": "Lip Sync",
            "score": result["lip_sync_mismatch_score"],
            "reason": "Lip-sync analysis classified the aggregate synchronization pattern as consistent.",
        })
    else:
        fake_support.append({
            "modality": "lip_sync",
            "label": "Lip Sync",
            "score": result["lip_sync_mismatch_score"],
            "reason": "Lip-sync analysis classified the aggregate synchronization pattern as inconsistent.",
        })

    final_verdict = "likely deepfake" if result.get("prediction") == "DEEPFAKE" else "likely real"
    contradicting = fake_support if result.get("prediction") == "REAL" else real_support
    aligned = real_support if result.get("prediction") == "REAL" else fake_support
    consistency = evidence_report.get("evidence_consistency", "UNKNOWN")
    disagreement = evidence_report.get("modality_disagreement", "UNKNOWN")

    unavailable = []
    audio_evidence = result.get("_audio_evidence") or {}
    if audio_evidence.get("status") == "global_only":
        unavailable.append("Audio classifier evidence is global unless localized audio windows are explicitly returned.")
    if not audio_evidence.get("envelope"):
        unavailable.append("Audio waveform evidence is unavailable for this request.")
    semantic = result.get("_semantic_metadata") or {}
    semantic_health = result.get("_feature_health", {}).get("semantic", {})
    if semantic_health.get("status") == "zero_embedding":
        unavailable.append("Semantic analysis was degraded because the embedding was invalid.")
    else:
        unavailable.append("Semantic classifier evidence is global unless timestamped transcript segments are explicitly tied to an analysis result.")
    unavailable.append("Visual artifact localization is unavailable; context frames are not proof of manipulation.")

    detected_mismatch = None
    if lip_sync_status == "Inconsistent":
        detected_mismatch = (
            "Audio-visual synchronization inconsistency detected (rule-based lip-sync mismatch score)."
        )
    elif blink_status == "Abnormal":
        detected_mismatch = "Abnormal eye-blink pattern detected (rule-based anomaly score)."

    modality_explanations = _modality_explanations(result, blink_status, lip_sync_status)
    return {
        "summary": (
            f"The fusion model predicts this video as {final_verdict}, but the modality evidence is mixed."
            if consistency in ("LOW", "MIXED")
            else f"The fusion model predicts this video as {final_verdict}, and the available modality evidence is broadly consistent."
        ),
        "key_findings": [
            f"Visual analysis produced a {result['visual_fake_probability']:.0%} fake probability.",
            f"Audio analysis produced a {result['audio_fake_probability']:.0%} fake probability.",
            f"Semantic analysis produced a {result['semantic_fake_probability']:.0%} fake probability.",
            f"Blink analysis produced a {result['blink_anomaly_score']:.0%} anomaly score and was classified as {blink_status}.",
            f"Lip-sync analysis produced a {result['lip_sync_mismatch_score']:.0%} mismatch score and was classified as {lip_sync_status}.",
        ],
        "supporting_signals": aligned,
        "contradicting_signals": contradicting,
        "cross_modal_interpretation": (
            f"Evidence consistency is {consistency.lower()} and modality disagreement is {disagreement.lower()}. "
            f"The final fusion score is independent from the displayed modality probabilities: the model consumes learned feature representations and can legitimately disagree with any one modality signal."
        ),
        "fusion_interpretation": (
            f"The fusion model assigns {result['final_fake_probability']:.1%} fake probability. "
            "This final output is computed from learned feature embeddings and cross-modal fusion, not by averaging the displayed modality probabilities."
            if evidence_report.get("review_recommended")
            else f"The fusion model assigns {result['final_fake_probability']:.1%} fake probability, and it can reasonably disagree with any single modality-level score because it combines multiple learned feature streams."
        ),
        "evidence_limitations": unavailable,
        "review_recommended": bool(evidence_report.get("review_recommended")),
        "confidence_level": evidence_report.get("confidence_level", "unknown"),
        "evidence_consistency": consistency,
        "modality_disagreement": disagreement,
        "detected_mismatch": detected_mismatch,
        "modality_explanations": modality_explanations,
        "explanation": (
            f"The fusion model predicts {final_verdict}. Evidence consistency is {consistency.lower()} "
            f"and modality disagreement is {disagreement.lower()}. The final fusion decision can disagree with individual modality signals because the model combines learned multimodal feature representations, not the displayed modality probabilities."
        ),
        "overall_interpretation": f"The fusion model combined the five modality inputs and classified this video as {final_verdict}. It is possible for the final fusion score to disagree with one or more modality-level scores because the fusion network learns cross-modal interactions from the underlying feature embeddings.",
        "semantic_context": {
            "language": semantic.get("language", "unknown"),
            "transcript": semantic.get("transcript", ""),
            "segments": semantic.get("segments", []),
            "confidence": semantic.get("confidence"),
            "reliable": semantic.get("reliable", False),
            "evidence_status": semantic_health.get("status", "unavailable"),
            "explanation": (
                "Semantic analysis was degraded because the generated embedding was invalid."
                if semantic_health.get("status") == "zero_embedding"
                else "Semantic classifier signal is global; no localized textual evidence is available."
            ),
        },
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
    if not isinstance(result, dict) or not result.get("evidence"):
        raise InferenceFailedError(
            "Inference returned an incomplete result.",
            details="The backend did not return the required scoring fields.",
        )
    evidence_report = result["evidence"]
    fps = meta.get("fps") or 0.0

    modalities = {
        "visual": {
            "label": MODALITY_LABELS["visual"],
            "kind": "learned_probability",
            "fake_probability": result["visual_fake_probability"],
            "evidence_status": "unavailable",
            "evidence": "Global classifier signal",
            "localization": "unavailable",
            "trust": "Treat as one learned signal; it does not identify a specific visual artifact.",
            "explanation": "The visual classifier found features associated with manipulated content, but the current pipeline cannot identify a specific visual artifact responsible for the score.",
            "feature_health": result.get("_feature_health", {}).get("visual"),
        },
        "audio": {
            "label": MODALITY_LABELS["audio"],
            "kind": "learned_probability",
            "fake_probability": result["audio_fake_probability"],
            "evidence_status": result.get("_audio_evidence", {}).get("status", "unavailable"),
            "evidence": "Global classifier signal",
            "localization": "localized" if result.get("_audio_evidence", {}).get("localized_windows") else "unavailable",
            "trust": "Treat as a global learned signal unless timestamped audio windows are present.",
            "explanation": "The audio classifier found characteristics associated with manipulated speech. This score alone does not identify a specific altered section unless timestamped evidence is available.",
            "feature_health": result.get("_feature_health", {}).get("audio"),
        },
        "semantic": {
            "label": MODALITY_LABELS["semantic"],
            "kind": "learned_probability",
            "fake_probability": result["semantic_fake_probability"],
            "evidence_status": (
                "degraded" if result.get("_feature_health", {}).get("semantic", {}).get("status") == "zero_embedding"
                else "global_only"
            ),
            "evidence": "Global semantic classifier signal",
            "localization": "unavailable",
            "trust": "Do not attribute manipulation to a transcript segment unless localized semantic evidence is returned.",
            "explanation": "The semantic classifier found content patterns associated with its fake class. No specific transcript segment should be called manipulated unless the analysis provides localized evidence.",
            "feature_health": result.get("_feature_health", {}).get("semantic"),
        },
        "eye_blink": {
            "label": "Eye Blink",
            "kind": "rule_based_score",
            "anomaly_score": result["blink_anomaly_score"],
            "status": result["blink_status"],
            "evidence_status": "localized" if result.get("_blink_events") else "not_detected",
            "evidence": "Rule-based blink statistics",
            "localization": "localized" if result.get("_blink_events") else "aggregate_only",
            "trust": "Blink anomalies are supporting signals and are not proof of manipulation by themselves.",
            "explanation": (
                "The measured blink pattern crossed the configured abnormality criteria."
                if result["blink_status"] == "Abnormal"
                else "The measured blink pattern did not cross the configured abnormality criteria."
            ),
            "feature_health": result.get("_feature_health", {}).get("blink"),
        },
        "lip_sync": {
            "label": "Lip Sync",
            "kind": "rule_based_score",
            "mismatch_score": result["lip_sync_mismatch_score"],
            "status": result["lip_sync_status"],
            "evidence_status": "localized" if isinstance(result.get("window_evidence"), list) and result.get("window_evidence") else "unavailable",
            "evidence": "Rule-based audio/visual timing score",
            "localization": "localized" if isinstance(result.get("window_evidence"), list) and result.get("window_evidence") else "aggregate_only",
            "trust": "Lip-sync mismatch is a consistency signal and is not proof of deepfake by itself.",
            "explanation": (
                "The current rule-based criteria classify the overall synchronization as inconsistent."
                if result["lip_sync_status"] == "Inconsistent"
                else "Some mouth/audio timing variation may be present, but the current rule-based criteria classify the overall synchronization as consistent."
            ),
            "feature_health": result.get("_feature_health", {}).get("lipsync"),
        },
    }

    evidence = {}
    frames = _build_frames(result.get("frame_evidence"), fps, job_id)
    if frames:
        evidence["frames"] = frames
    audio_evidence = result.get("_audio_evidence")
    if audio_evidence and audio_evidence.get("envelope"):
        audio_evidence = dict(audio_evidence)
        audio_evidence["media_url"] = f"/api/media/{job_id}"
        evidence["audio_waveform"] = audio_evidence
    elif audio_evidence:
        evidence["audio_evidence_status"] = audio_evidence.get("status", "unavailable")
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
    if result.get("_blink_events"):
        evidence["blink_events"] = [
            {"start_frame": start, "end_frame": end,
             "start_seconds": round(start / fps, 3) if fps else None,
             "end_seconds": round(end / fps, 3) if fps else None,
             "evidence_status": "supporting_context"}
            for start, end in result["_blink_events"]
        ]
    public_health = {
        name: {key: value for key, value in health.items() if key != "path"}
        for name, health in result.get("_feature_health", {}).items()
    }

    return {
        "video_filename": video_filename,
        "video_duration_seconds": meta.get("duration_seconds"),
        "analyzed_at": analyzed_at,
        "processing_time_seconds": round(processing_time_seconds, 3),
        "final_verdict": "LIKELY_DEEPFAKE" if result["prediction"] == "DEEPFAKE" else "LIKELY_REAL",
        "final_fake_probability": result["final_fake_probability"],
        "final_real_probability": result["final_real_probability"],
        "low_confidence": evidence_report.get("low_confidence"),
        "evidence_consistency": evidence_report.get("evidence_consistency"),
        "modality_disagreement": evidence_report.get("modality_disagreement"),
        "review_recommended": evidence_report.get("review_recommended"),
        "confidence_level": evidence_report.get("confidence_level"),
        "model_confidence_note": MODEL_CONFIDENCE_NOTE,
        "modalities": modalities,
        "attention_summary": {
            "weights": _rename_modality_keys(result["attention_summary"]),
            "note": ATTENTION_NOTE,
        },
        "modality_contributions": {
            "deltas": _rename_modality_keys(result["modality_contributions"].get("contributions", {})),
            "ablated_fake_probabilities": _rename_modality_keys(
                result["modality_contributions"].get("ablated_fake_probabilities", {})
            ),
            "note": CONTRIBUTION_NOTE,
        },
        "fusion_diagnostics": result.get("fusion_diagnostics"),
        "evidence": evidence,
        "feature_health": public_health,
        "contextual_analysis": _build_contextual_analysis(
            result, evidence_report, result["lip_sync_status"], result["blink_status"]
        ),
    }
