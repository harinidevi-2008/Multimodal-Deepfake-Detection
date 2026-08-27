"""
Evaluate the rule-based blink and lip-sync scores against ground-truth
real/fake labels, on one held-out split from fusion/data_split.json.

Important distinction: blink and lip-sync scores come from fixed,
hand-designed rules (EAR thresholds, correlation) - not a trained
classifier - there is no learned decision boundary. These are reported
as ROC-AUC/accuracy/precision/recall/F1 like a classifier's numbers
for comparability, but must stay clearly labeled RULE-BASED and must
never be called "probabilities" (see eval/generate_report.py's "D"
section, which does exactly that labeling).

Classification thresholds:
  - Blink: predicted "Abnormal" exactly when
    blink_irregularity_score >= BLINK_ANOMALY_THRESHOLD (0.5) - the
    same comparison blink_analyzer.py's own analyze_blinks() uses.
  - Lip-sync: predicted "Inconsistent" is derived by calling
    lipsync_analyzer.py's OWN lipsync_status_from_consistency() on the
    precomputed consistency value (sync_score in the feature vector),
    rather than re-deriving an equivalent threshold in mismatch-score
    space. That reconstruction previously had an off-by-one boundary
    bug: at consistency == CONSISTENCY_THRESHOLD exactly, the analyzer
    calls it "Consistent" (>=), but `mismatch_score >= 1 - threshold`
    flagged that same sample as "Inconsistent" - the boundary sample
    disagreed with the analyzer that produced it. Reusing the
    analyzer's own function eliminates that whole class of bug by
    construction. See fusion/test_lipsync_consistency_boundary.py.

Usage:
    python fusion/evaluate_blink_lipsync.py \
        --blink-root visual/data/blink_features \
        --lipsync-root visual/data/lipsync_features \
        --split test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import DEFAULT_SPLIT_PATH, load_split, split_of  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visual" / "src" / "lipsync"))
from lipsync_analyzer import CONSISTENCY_THRESHOLD, lipsync_status_from_consistency  # noqa: E402

# Must match blink_analyzer.py's own inline threshold (0.5). Kept as a
# named constant here since blink_analyzer.py doesn't export one itself
# (unlike lipsync, there is no boundary-direction bug for blink - both
# this constant and the analyzer use the same ">=" comparison - so no
# refactor of blink_analyzer.py was needed; see the correction report).
BLINK_ANOMALY_THRESHOLD = 0.5


def load_vectors_and_labels(feature_root, split_data=None, split_name="test"):
    """
    Loads every .npy vector under feature_root, optionally filtered to
    one persisted split (split_data=None means "all", no filtering).
    Returns (vectors: np.ndarray shape (n, dim), labels: np.ndarray).
    """
    feature_root = Path(feature_root)
    vectors, labels = [], []

    for npy_file in sorted(feature_root.rglob("*.npy")):
        relative_path = npy_file.relative_to(feature_root)
        first_folder = relative_path.parts[0]

        if first_folder.startswith("RealVideo"):
            label = 0
        elif first_folder.startswith("FakeVideo"):
            label = 1
        else:
            continue

        if split_data is not None and split_name != "all":
            if split_of(relative_path, split_data) != split_name:
                continue

        vectors.append(np.load(npy_file))
        labels.append(label)

    if not vectors:
        return np.zeros((0,)), np.array([])
    return np.stack(vectors), np.array(labels)


def _metrics(score_name, scores, labels, preds, split, threshold, threshold_note=None):
    try:
        roc_auc = roc_auc_score(labels, scores)
    except ValueError:
        roc_auc = float("nan")

    return {
        "score_name": score_name,
        "split": split,
        "n_samples": int(len(labels)),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "threshold": threshold,
        "threshold_note": threshold_note,
        "roc_auc": roc_auc,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }


def evaluate_blink(blink_root, split="test", split_path=DEFAULT_SPLIT_PATH):
    """Returns a metrics dict, or None if no samples were found for this split."""
    split_data = load_split(split_path) if split != "all" else None
    vectors, labels = load_vectors_and_labels(blink_root, split_data, split)
    if len(labels) == 0:
        return None

    # blink vector layout: [blink_count, blink_rate_per_min, avg_blink_duration_sec, blink_irregularity_score]
    scores = vectors[:, 3]
    preds = (scores >= BLINK_ANOMALY_THRESHOLD).astype(int)
    return _metrics("blink_anomaly_score", scores, labels, preds, split, BLINK_ANOMALY_THRESHOLD)


def evaluate_lipsync(lipsync_root, split="test", split_path=DEFAULT_SPLIT_PATH):
    """Returns a metrics dict, or None if no samples were found for this split."""
    split_data = load_split(split_path) if split != "all" else None
    vectors, labels = load_vectors_and_labels(lipsync_root, split_data, split)
    if len(labels) == 0:
        return None

    # lipsync vector layout: [sync_score (= consistency), mismatch_score]
    consistency = vectors[:, 0]
    mismatch = vectors[:, 1]

    preds = np.array(
        [1 if lipsync_status_from_consistency(c) == "Inconsistent" else 0 for c in consistency]
    )
    return _metrics(
        "lip_sync_mismatch_score", mismatch, labels, preds, split, CONSISTENCY_THRESHOLD,
        threshold_note=f"classification derived from lipsync_analyzer.CONSISTENCY_THRESHOLD="
                        f"{CONSISTENCY_THRESHOLD} applied to consistency via lipsync_status_from_consistency(), "
                        f"not a separately maintained mismatch-space threshold",
    )


def _print_result(label, result):
    if result is None:
        print(f"\n=== {label}: no samples found ===")
        return
    print(f"\n=== {label} ({result['n_samples']} samples, split='{result['split']}', "
          f"real={result['n_real']} fake={result['n_fake']}, RULE-BASED, "
          f"fixed threshold={result['threshold']}) ===")
    if result["threshold_note"]:
        print(f"  ({result['threshold_note']})")
    print(f"ROC-AUC:   {result['roc_auc']:.4f}  (threshold-independent ranking quality)")
    print(f"Accuracy:  {result['accuracy']:.4f}")
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1:        {result['f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate rule-based blink and lip-sync scores on a held-out split.")
    parser.add_argument("--blink-root", required=True)
    parser.add_argument("--lipsync-root", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"],
                         help="Defaults to 'test' (held-out). 'all' is NOT a held-out evaluation - debugging only.")
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    args = parser.parse_args()

    if args.split == "all":
        print("WARNING: --split all evaluates on every available sample regardless of train/val/test "
              "membership. This is NOT a held-out evaluation - use only for debugging.\n")

    blink_result = evaluate_blink(args.blink_root, split=args.split, split_path=args.split_path)
    lipsync_result = evaluate_lipsync(args.lipsync_root, split=args.split, split_path=args.split_path)

    _print_result("BLINK anomaly score (rule-based, higher = more abnormal)", blink_result)
    _print_result("LIP-SYNC mismatch score (rule-based, higher = more inconsistent)", lipsync_result)

    if blink_result is None or lipsync_result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
