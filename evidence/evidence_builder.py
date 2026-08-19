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

Important distinction (see project notes): blink_anomaly and
lipsync_mismatch are rule-based scores, not outputs of a trained
classifier - they're included as supporting evidence, not treated as
equivalent to the three learned modality probabilities.
"""

# Thresholds above which a signal is called out as a contributing reason.
VISUAL_PROB_THRESHOLD = 0.6
AUDIO_PROB_THRESHOLD = 0.6
SEMANTIC_PROB_THRESHOLD = 0.6

# Final verdict banding.
FINAL_PROB_HIGH = 0.75
FINAL_PROB_MEDIUM = 0.5


def _verdict_label(final_prob):
    if final_prob >= FINAL_PROB_HIGH:
        return "Likely Deepfake"
    if final_prob >= FINAL_PROB_MEDIUM:
        return "Possibly Manipulated"
    return "Likely Authentic"


def build_evidence_report(
    visual_probability,
    audio_probability,
    semantic_probability,
    blink_result,
    lipsync_result,
    final_fake_probability,
):
    """
    Parameters:
        visual_probability (float): P(fake) from classifiers/evaluate_classifier.py (modality=visual)
        audio_probability (float): P(fake) from classifiers/evaluate_classifier.py (modality=audio)
        semantic_probability (float): P(fake) from classifiers/evaluate_classifier.py (modality=semantic)
        blink_result (dict): output of eye_blink/blink_analyzer.py's analyze_blinks()
        lipsync_result (dict): output of lipsync/lipsync_analyzer.py's analyze_lipsync()
        final_fake_probability (float): P(fake) from the fusion model (original or enhanced)

    Returns:
        dict matching the target evidence-report schema, including a
        plain-language "evidence" list and a "verdict" label.
    """

    reasons = []

    if visual_probability >= VISUAL_PROB_THRESHOLD:
        reasons.append("Visual stream shows high fake probability")

    if audio_probability >= AUDIO_PROB_THRESHOLD:
        reasons.append("Audio stream shows high fake probability")

    if semantic_probability >= SEMANTIC_PROB_THRESHOLD:
        reasons.append("Semantic stream shows high fake probability")

    if blink_result.get("blink_status") == "Abnormal":
        reasons.append("Abnormal blinking pattern")

    if lipsync_result.get("sync_status") == "Inconsistent":
        reasons.append("Audio-visual synchronization inconsistency")

    if not reasons:
        reasons.append("No individual signal exceeded its threshold - low-confidence result")

    return {
        "visual_probability": round(float(visual_probability), 4),
        "audio_probability": round(float(audio_probability), 4),
        "semantic_probability": round(float(semantic_probability), 4),
        "blink_anomaly": blink_result.get("blink_anomaly_score", blink_result.get("blink_irregularity_score")),
        "blink_status": blink_result.get("blink_status"),
        "lip_sync_mismatch": lipsync_result.get("mismatch_score"),
        "lip_sync_status": lipsync_result.get("sync_status"),
        "final_fake_probability": round(float(final_fake_probability), 4),
        "verdict": _verdict_label(final_fake_probability),
        "evidence": reasons,
    }
