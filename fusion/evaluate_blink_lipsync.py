"""
Evaluate the rule-based blink and lip-sync scores against ground-truth
real/fake labels.

Important distinction (flagged in the project notes): blink and
lip-sync scores come from fixed, hand-designed rules (EAR thresholds,
correlation), not a trained classifier - there is no learned decision
boundary and no train/val split here. Reporting an "accuracy" for them
the same way as classifiers/evaluate_classifier.py would misleadingly
imply they were fit to this data. This script instead reports:

  - ROC-AUC of the raw scalar score against the true label, which only
    asks "does this score rank fakes above reals?" without needing a
    threshold at all.
  - Accuracy/precision/recall/F1 at the SAME fixed threshold the
    analyzer itself already uses to decide Normal/Abnormal or
    Consistent/Inconsistent (0.5 for blink, 0.40 for lip-sync) - not a
    threshold tuned on this data, so this is a fair test of the rule as
    designed, not an optimistic best-case number.

Usage:
    python fusion/evaluate_blink_lipsync.py \
        --blink-root visual/data/blink_features \
        --lipsync-root visual/data/lipsync_features
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Must match blink_analyzer.py / lipsync_analyzer.py's own thresholds.
BLINK_ANOMALY_THRESHOLD = 0.5
LIPSYNC_MISMATCH_THRESHOLD = 0.6  # mismatch_score = 1 - consistency; consistency < 0.40 -> mismatch > 0.60


def load_scores_and_labels(feature_root, score_index):
    feature_root = Path(feature_root)
    scores, labels = [], []

    for npy_file in sorted(feature_root.rglob("*.npy")):
        relative_path = npy_file.relative_to(feature_root)
        first_folder = relative_path.parts[0]

        if first_folder.startswith("RealVideo"):
            label = 0
        elif first_folder.startswith("FakeVideo"):
            label = 1
        else:
            continue

        vector = np.load(npy_file)
        scores.append(float(vector[score_index]))
        labels.append(label)

    return np.array(scores), np.array(labels)


def report(name, scores, labels, threshold):
    if len(scores) == 0:
        print(f"\n=== {name}: no samples found ===")
        return

    preds = (scores >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(labels, scores)
    except ValueError:
        roc_auc = float("nan")

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    print(f"\n=== {name} ({len(scores)} samples, rule-based, fixed threshold={threshold}) ===")
    print(f"ROC-AUC:   {roc_auc:.4f}  (threshold-independent ranking quality)")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate rule-based blink and lip-sync scores.")
    parser.add_argument("--blink-root", required=True)
    parser.add_argument("--lipsync-root", required=True)
    args = parser.parse_args()

    # blink vector layout: [blink_count, blink_rate_per_min, avg_blink_duration_sec, blink_irregularity_score]
    blink_scores, blink_labels = load_scores_and_labels(args.blink_root, score_index=3)
    report("BLINK anomaly score", blink_scores, blink_labels, BLINK_ANOMALY_THRESHOLD)

    # lipsync vector layout: [sync_score, mismatch_score]
    lipsync_scores, lipsync_labels = load_scores_and_labels(args.lipsync_root, score_index=1)
    report("LIP-SYNC mismatch score", lipsync_scores, lipsync_labels, LIPSYNC_MISMATCH_THRESHOLD)


if __name__ == "__main__":
    main()
