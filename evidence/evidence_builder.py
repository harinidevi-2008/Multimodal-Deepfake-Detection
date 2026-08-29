"""
Evidence / explanation layer.

Combines the separate outputs of the visual/audio/semantic classifiers,
the (rule-based) blink and lip-sync analyzers, and the fusion model's
final probability into one structured report with plain-language
reasons - turning a bare "DEEPFAKE 93%" into something that says why.

This is a pure composition function: it does not run any model itself.
It takes whatever scores you already computed elsewhere (classifiers/
evaluate_classifier.py, fusion/evaluate_fusion.py or
evaluate_enhanced_fusion.py, eye_blink/blink_analyzer.py,
lipsync/lipsync_analyzer.py) and assembles them. Keeping it decoupled
this way means it works whether you're using the original 3-modality
fusion model or the enhanced 5-modality one, and doesn't need to import
any of the model code.

TERMINOLOGY (important, and the reason for a breaking rename in this
pass): visual/audio/semantic scores are outputs of TRAINED classifiers
and are legitimately called "fake probabilities". blink_result and
lipsync_result come from fixed, hand-designed rules (EAR thresholds,
mouth/audio correlation) with no learned decision boundary - calling
those "probabilities" would misleadingly imply they were fit to data.
This module's parameter names and output keys now say exactly that:
"*_fake_probability" for the three learned classifiers, and
"blink_anomaly_score" / "lip_sync_mismatch_score" for the two
rule-based signals.

BREAKING CHANGE from the previous version of this file: parameters
were renamed visual_probability -> visual_fake_probability (and
audio/semantic likewise); output keys blink_anomaly -> blink_anomaly_score
and lip_sync_mismatch -> lip_sync_mismatch_score; the lip-sync-status
lookup was switched from lipsync_result.get("sync_status") to
lipsync_result.get("lipsync_status") (sync_status was a duplicate key
being removed from lipsync_analyzer.py's output - see that module).
This function has exactly one caller in this repository
(evidence_test.py, updated alongside this file) so the blast radius of
the rename is contained to this module and its own demo/test script.
"""

# Thresholds above which a signal is called out as a contributing reason.
VISUAL_PROB_THRESHOLD = 0.6
AUDIO_PROB_THRESHOLD = 0.6
SEMANTIC_PROB_THRESHOLD = 0.6

# Final verdict banding.
FINAL_PROB_HIGH = 0.75
FINAL_PROB_MEDIUM = 0.5

# A final_fake_probability within this margin of the 0.5 decision
# boundary is the ONLY thing this module calls "low confidence" - it is
# a property of the fused decision itself, not a consequence of no
# individual stream crossing its own threshold (the 5-modal model can
# and does make confident decisions from combined evidence even when no
# single stream individually crosses a threshold - see
# build_evidence_report()'s no-individual-signal wording below).
LOW_CONFIDENCE_MARGIN = 0.15
LEARNED_SIGNAL_THRESHOLD = 0.6

LEARNED_MODALITIES = ("visual", "audio", "semantic")


def _verdict_label(final_prob):
    if final_prob >= FINAL_PROB_HIGH:
        return "Likely Deepfake"
    if final_prob >= FINAL_PROB_MEDIUM:
        return "Possibly Manipulated"
    return "Likely Authentic"


def _is_low_confidence(final_prob):
    return abs(final_prob - 0.5) <= LOW_CONFIDENCE_MARGIN


def _signal_strength(value):
    value = float(value)
    if value >= 0.75:
        return "strong fake signal"
    if value >= LEARNED_SIGNAL_THRESHOLD:
        return "moderate fake signal"
    if value <= 0.25:
        return "strong real signal"
    if value < 0.4:
        return "moderate real signal"
    return "mixed/weak signal"


def _build_learned_signals(visual_fake_probability, audio_fake_probability, semantic_fake_probability):
    scores = {
        "visual": float(visual_fake_probability),
        "audio": float(audio_fake_probability),
        "semantic": float(semantic_fake_probability),
    }
    signals = []
    for name in LEARNED_MODALITIES:
        score = scores[name]
        supports = "fake" if score >= 0.5 else "real"
        signals.append({
            "modality": name,
            "score": round(score, 4),
            "score_type": "classifier_fake_probability",
            "supports": supports,
            "strength": _signal_strength(score),
            "localized": False,
            "evidence_status": "global_only",
        })
    return signals


def _build_rule_signals(blink_result, lipsync_result):
    blink_score = float(blink_result.get("blink_irregularity_score") or 0.0)
    lipsync_score = float(lipsync_result.get("mismatch_score") or 0.0)
    return [
        {
            "modality": "blink",
            "score": round(blink_score, 4),
            "score_type": "rule_based_anomaly_score",
            "status": blink_result.get("blink_status"),
            "supports": "fake" if blink_result.get("blink_status") == "Abnormal" else "real",
            "strength": "abnormal rule-based signal" if blink_result.get("blink_status") == "Abnormal" else "no abnormal rule-based signal",
            "localized": bool(blink_result.get("blink_events")),
            "evidence_status": "localized" if blink_result.get("blink_events") else "aggregate_only",
        },
        {
            "modality": "lipsync",
            "score": round(lipsync_score, 4),
            "score_type": "rule_based_mismatch_score",
            "status": lipsync_result.get("lipsync_status"),
            "supports": "fake" if lipsync_result.get("lipsync_status") == "Inconsistent" else "real",
            "strength": "inconsistent rule-based signal" if lipsync_result.get("lipsync_status") == "Inconsistent" else "no inconsistent rule-based signal",
            "localized": bool(lipsync_result.get("window_evidence")),
            "evidence_status": "localized" if lipsync_result.get("window_evidence") else "aggregate_only",
        },
    ]


def _build_consistency(final_fake_probability, signals):
    final_support = "fake" if final_fake_probability >= 0.5 else "real"
    strong_opposing = [
        signal for signal in signals
        if signal["supports"] != final_support
        and (
            (signal["score_type"] == "classifier_fake_probability" and signal["score"] >= LEARNED_SIGNAL_THRESHOLD)
            or signal.get("status") in ("Abnormal", "Inconsistent")
        )
    ]
    aligned = [signal for signal in signals if signal["supports"] == final_support]
    if len(strong_opposing) >= 2:
        level = "LOW"
        disagreement = "HIGH"
        review_recommended = True
    elif len(strong_opposing) == 1:
        level = "MIXED"
        disagreement = "MODERATE"
        review_recommended = True
    else:
        level = "HIGH"
        disagreement = "LOW"
        review_recommended = False
    if _is_low_confidence(final_fake_probability):
        review_recommended = True
        if level == "HIGH":
            level = "MIXED"
    return {
        "evidence_consistency": level,
        "modality_disagreement": disagreement,
        "review_recommended": review_recommended,
        "final_support": final_support,
        "aligned_signal_count": len(aligned),
        "contradicting_signal_count": len(strong_opposing),
        "contradicting_modalities": [signal["modality"] for signal in strong_opposing],
        "confidence_level": "low" if review_recommended and level == "LOW" else ("medium" if review_recommended else "high"),
    }


def build_evidence_report(
    visual_fake_probability,
    audio_fake_probability,
    semantic_fake_probability,
    blink_result,
    lipsync_result,
    final_fake_probability,
):
    """
    Parameters:
        visual_fake_probability (float): P(fake) from classifiers/evaluate_classifier.py (modality=visual)
        audio_fake_probability (float): P(fake) from classifiers/evaluate_classifier.py (modality=audio)
        semantic_fake_probability (float): P(fake) from classifiers/evaluate_classifier.py (modality=semantic)
        blink_result (dict): output of eye_blink/blink_analyzer.py's analyze_blinks() -
            a RULE-BASED anomaly score, not a learned probability.
        lipsync_result (dict): output of lipsync/lipsync_analyzer.py's analyze_lipsync() -
            a RULE-BASED mismatch score, not a learned probability.
        final_fake_probability (float): P(fake) from the fusion model (original or enhanced)

    Returns:
        dict matching the target evidence-report schema, including a
        plain-language "evidence" list and a "verdict" label.
    """

    reasons = []

    if visual_fake_probability >= VISUAL_PROB_THRESHOLD:
        reasons.append("Visual stream shows high fake probability")

    if audio_fake_probability >= AUDIO_PROB_THRESHOLD:
        reasons.append("Audio stream shows high fake probability")

    if semantic_fake_probability >= SEMANTIC_PROB_THRESHOLD:
        reasons.append("Semantic stream shows high fake probability")

    if blink_result.get("blink_status") == "Abnormal":
        reasons.append("Abnormal blinking pattern (rule-based anomaly score)")

    if lipsync_result.get("lipsync_status") == "Inconsistent":
        reasons.append("Audio-visual synchronization inconsistency (rule-based mismatch score)")

    if not reasons:
        # IMPORTANT: this does NOT mean the result is low-confidence. The
        # 5-modal fusion model combines all five streams through learned
        # cross-attention and can reach a confident decision even when no
        # single stream individually crosses its own manually-chosen
        # threshold - that's the point of fusing evidence rather than
        # voting on individual thresholds. Whether this particular result
        # is actually low-confidence is a separate, explicit check below,
        # based on the final probability itself.
        reasons.append(
            "No individual signal exceeded its evidence threshold; the final result is based on "
            "combined multimodal evidence."
        )

    signals = (
        _build_learned_signals(visual_fake_probability, audio_fake_probability, semantic_fake_probability)
        + _build_rule_signals(blink_result, lipsync_result)
    )
    consistency = _build_consistency(float(final_fake_probability), signals)

    low_confidence = _is_low_confidence(final_fake_probability)
    if low_confidence:
        reasons.append(
            f"Final probability is within {LOW_CONFIDENCE_MARGIN:.2f} of the 0.5 decision boundary - "
            "genuinely low-confidence result."
        )

    return {
        "visual_fake_probability": round(float(visual_fake_probability), 4),
        "audio_fake_probability": round(float(audio_fake_probability), 4),
        "semantic_fake_probability": round(float(semantic_fake_probability), 4),
        "blink_anomaly_score": blink_result.get("blink_irregularity_score"),
        "blink_status": blink_result.get("blink_status"),
        "lip_sync_mismatch_score": lipsync_result.get("mismatch_score"),
        "lip_sync_status": lipsync_result.get("lipsync_status"),
        "final_fake_probability": round(float(final_fake_probability), 4),
        "verdict": _verdict_label(final_fake_probability),
        "low_confidence": low_confidence,
        "evidence": reasons,
        "signals": signals,
        **consistency,
    }
