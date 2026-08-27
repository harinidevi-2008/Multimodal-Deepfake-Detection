"""
Feature-quality audit: a read-only inspection of the five precomputed
feature roots before real training, to catch corrupted/malformed files
early rather than discovering them mid-training-run (a NaN loss after
an hour of training is a much more expensive way to find the same
problem).

Does NOT modify any feature file. Does NOT re-extract anything.

Usage:
    python fusion/audit_features.py \
        --visual-root visual/data/features_aligned --audio-root audio/data/features \
        --semantic-root semantic/data/features --blink-root visual/data/blink_features \
        --lipsync-root visual/data/lipsync_features
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_ROOT,
    DEFAULT_BLINK_ROOT,
    DEFAULT_LIPSYNC_ROOT,
    DEFAULT_SEMANTIC_ROOT,
    DEFAULT_VISUAL_ROOT,
)

EXPECTED_DIMS = {"visual": 1280, "audio": 768, "semantic": 384, "blink": 4, "lipsync": 2}

# If the zero-vector rate differs by at least this many percentage
# points between Real and Fake, flag it for investigation.
ZERO_VECTOR_IMBALANCE_FLAG_POINTS = 15.0


def _label_from_relative_path(relative_path):
    first_folder = relative_path.parts[0]
    if first_folder.startswith("RealVideo"):
        return 0
    if first_folder.startswith("FakeVideo"):
        return 1
    return None


def audit_root(name, root, expected_dim):
    root = Path(root)
    files = sorted(root.rglob("*.npy"))
    expected_shape = (expected_dim,)

    count = 0
    shape_counts = {}
    dtype_counts = {}
    nan_count = 0
    inf_count = 0
    unexpected_shape_count = 0
    zero_vector_by_label = {0: 0, 1: 0, None: 0}
    label_counts = {0: 0, 1: 0, None: 0}
    unreadable = []

    for f in files:
        try:
            arr = np.load(f)
        except Exception as exc:  # noqa: BLE001 - report and keep auditing the rest
            unreadable.append((str(f), str(exc)))
            continue

        count += 1
        shape_counts[arr.shape] = shape_counts.get(arr.shape, 0) + 1
        dtype_counts[str(arr.dtype)] = dtype_counts.get(str(arr.dtype), 0) + 1
        nan_count += int(np.isnan(arr).sum())
        inf_count += int(np.isinf(arr).sum())
        # Direct tuple comparison, not a string comparison - (1, 1280)
        # must be caught as different from (1280,) regardless of how
        # either shape happens to be formatted for display below. This
        # audit only detects bad shapes; it never reshapes/repairs them.
        if arr.shape != expected_shape:
            unexpected_shape_count += 1

        relative_path = f.relative_to(root)
        label = _label_from_relative_path(relative_path)
        label_counts[label] = label_counts.get(label, 0) + 1

        if bool(np.all(arr == 0)):
            zero_vector_by_label[label] = zero_vector_by_label.get(label, 0) + 1

    return {
        "name": name,
        "root": str(root),
        "expected_dim": expected_dim,
        "expected_shape": expected_shape,
        "count": count,
        "shapes": {str(k): v for k, v in shape_counts.items()},
        "dtypes": dtype_counts,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "unexpected_shape_count": unexpected_shape_count,
        "zero_vector_by_label": {
            "real": zero_vector_by_label.get(0, 0),
            "fake": zero_vector_by_label.get(1, 0),
            "unknown": zero_vector_by_label.get(None, 0),
        },
        "label_counts": {
            "real": label_counts.get(0, 0), "fake": label_counts.get(1, 0), "unknown": label_counts.get(None, 0),
        },
        "unreadable": unreadable,
    }


def collect_relative_paths(root):
    root = Path(root)
    return {str(f.relative_to(root)) for f in root.rglob("*.npy")}


def check_alignment(roots, max_report=20):
    """
    Read-only cross-root check: verifies that the SAME relative paths
    exist across all five feature roots. Each audit_root() call above
    only inspects a file's own contents (shape/NaN/Inf) - it says
    nothing about whether that file's counterpart exists in the OTHER
    four roots. A sample missing from even one root is silently
    excluded by EnhancedFusionDataset / create_split.py's
    collect_aligned_samples() (by design - that is the correct
    behavior, not a bug), so misalignment is expected during
    incremental extraction rather than necessarily a defect - but it
    should be visible here, not only discovered later as a
    smaller-than-expected training set.

    roots: {name: root_path} for all five modalities.

    Returns per_root_relative_path_counts, total_distinct_relative_paths,
    fully_aligned_count, missing_from_at_least_one_root (up to
    max_report {"relative_path", "missing_from"} entries), and
    total_missing_from_at_least_one_root (the true count, even when the
    listed sample is truncated to max_report).
    """
    per_root_paths = {name: collect_relative_paths(root) for name, root in roots.items()}
    all_paths = set()
    for paths in per_root_paths.values():
        all_paths |= paths

    fully_aligned = 0
    missing = []
    total_missing = 0
    for rp in sorted(all_paths):
        missing_from = [name for name, paths in per_root_paths.items() if rp not in paths]
        if missing_from:
            total_missing += 1
            if len(missing) < max_report:
                missing.append({"relative_path": rp, "missing_from": missing_from})
        else:
            fully_aligned += 1

    return {
        "per_root_relative_path_counts": {name: len(paths) for name, paths in per_root_paths.items()},
        "total_distinct_relative_paths": len(all_paths),
        "fully_aligned_count": fully_aligned,
        "missing_from_at_least_one_root": missing,
        "total_missing_from_at_least_one_root": total_missing,
    }


def _print_stream_report(report):
    print(f"\n=== {report['name'].upper()} ({report['root']}) ===")
    print(f"Files readable: {report['count']}")
    if report["unreadable"]:
        print(f"UNREADABLE FILES: {len(report['unreadable'])} (first 5 shown)")
        for path, err in report["unreadable"][:5]:
            print(f"  {path}: {err}")
    print(f"Shapes: {report['shapes']}")
    print(f"Dtypes: {report['dtypes']}")
    print(f"NaN count (summed across all files): {report['nan_count']}")
    print(f"Inf count (summed across all files): {report['inf_count']}")
    print(f"Label counts: real={report['label_counts']['real']} fake={report['label_counts']['fake']} "
          f"unknown={report['label_counts']['unknown']}")

    if report["unexpected_shape_count"]:
        print(f"WARNING: {report['unexpected_shape_count']} file(s) with unexpected shape "
              f"(expected {report['expected_shape']}): {report['shapes']}")
    if report["nan_count"] or report["inf_count"]:
        print("WARNING: NaN/Inf values found - these will break training (loss becomes NaN) if not addressed.")


def _critical_problems(report):
    """
    The hard pre-training gate criteria: unreadable files, NaN, Inf, or
    an unexpected feature shape/dimension. Deliberately does NOT include
    zero-vector counts or the semantic zero-vector Real/Fake imbalance -
    those remain a printed flag for investigation, never a failure, per
    semantic/precompute.py's intentional zero-vectoring of unreliable
    transcripts.
    """
    problems = []
    if report["unreadable"]:
        problems.append(f"{report['name']}: {len(report['unreadable'])} unreadable .npy file(s)")
    if report["nan_count"]:
        problems.append(f"{report['name']}: {report['nan_count']} NaN value(s)")
    if report["inf_count"]:
        problems.append(f"{report['name']}: {report['inf_count']} Inf value(s)")
    if report["unexpected_shape_count"]:
        problems.append(
            f"{report['name']}: {report['unexpected_shape_count']} file(s) with unexpected shape "
            f"(expected {report['expected_shape']})"
        )
    return problems


def main():
    parser = argparse.ArgumentParser(description="Audit feature-file quality across all five modality roots.")
    parser.add_argument("--visual-root", default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--semantic-root", default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--blink-root", default=DEFAULT_BLINK_ROOT)
    parser.add_argument("--lipsync-root", default=DEFAULT_LIPSYNC_ROOT)
    args = parser.parse_args()

    reports = [
        audit_root("visual", args.visual_root, EXPECTED_DIMS["visual"]),
        audit_root("audio", args.audio_root, EXPECTED_DIMS["audio"]),
        audit_root("semantic", args.semantic_root, EXPECTED_DIMS["semantic"]),
        audit_root("blink", args.blink_root, EXPECTED_DIMS["blink"]),
        audit_root("lipsync", args.lipsync_root, EXPECTED_DIMS["lipsync"]),
    ]

    for r in reports:
        _print_stream_report(r)

    semantic_report = next(r for r in reports if r["name"] == "semantic")
    real_zero = semantic_report["zero_vector_by_label"]["real"]
    fake_zero = semantic_report["zero_vector_by_label"]["fake"]
    real_total = semantic_report["label_counts"]["real"]
    fake_total = semantic_report["label_counts"]["fake"]
    real_zero_pct = (real_zero / real_total * 100) if real_total else 0.0
    fake_zero_pct = (fake_zero / fake_total * 100) if fake_total else 0.0

    print("\n=== SEMANTIC zero-vector correlation with label ===")
    print("(semantic/precompute.py intentionally writes an all-zero vector when the transcript is "
          "unreliable - this checks only whether that's correlated with Real/Fake, it is not a bug by "
          "itself, and this script does NOT change that behavior.)")
    print(f"Real: {real_zero}/{real_total} zero vectors ({real_zero_pct:.1f}%)")
    print(f"Fake: {fake_zero}/{fake_total} zero vectors ({fake_zero_pct:.1f}%)")
    if abs(real_zero_pct - fake_zero_pct) >= ZERO_VECTOR_IMBALANCE_FLAG_POINTS:
        print(f"FLAG: zero-vector rate differs by >= {ZERO_VECTOR_IMBALANCE_FLAG_POINTS:.0f} percentage points "
              "between Real and Fake. The semantic stream may be partly learning 'was transcription "
              "reliable' instead of 'is this fake' - worth investigating before or during training. "
              "Not changed automatically. This is a WARNING only - it does NOT fail the audit.")
    else:
        print(f"No strong imbalance detected at the {ZERO_VECTOR_IMBALANCE_FLAG_POINTS:.0f}-point threshold used here.")

    alignment = check_alignment({
        "visual": args.visual_root, "audio": args.audio_root, "semantic": args.semantic_root,
        "blink": args.blink_root, "lipsync": args.lipsync_root,
    })
    print("\n=== CROSS-ROOT ALIGNMENT ===")
    print("(read-only - a sample missing from any root is excluded by create_split.py / "
          "EnhancedFusionDataset, not repaired here.)")
    print(f"Distinct relative paths seen across all roots: {alignment['total_distinct_relative_paths']}")
    print(f"Fully aligned (present in all five roots):     {alignment['fully_aligned_count']}")
    if alignment["total_missing_from_at_least_one_root"]:
        print(f"Missing from at least one root:                {alignment['total_missing_from_at_least_one_root']} "
              f"(showing up to {len(alignment['missing_from_at_least_one_root'])})")
        for m in alignment["missing_from_at_least_one_root"]:
            print(f"  {m['relative_path']}: missing from {m['missing_from']}")
    else:
        print("Every relative path found is present in all five feature roots.")

    # --- Hard pre-training gate ---
    # Critical == unreadable files / NaN / Inf / unexpected shape, on any
    # stream. Zero-vector counts and the semantic zero-vector imbalance
    # above are intentionally excluded - see _critical_problems()'s
    # docstring. This is what makes `python fusion/audit_features.py`
    # safe to use as the final pre-training gate: PASS means none of
    # those four problems were found; FAIL means at least one was.
    all_critical_problems = []
    for r in reports:
        all_critical_problems.extend(_critical_problems(r))

    print(f"\n{'=' * 60}")
    if all_critical_problems:
        print("FEATURE AUDIT: FAIL")
        for problem in all_critical_problems:
            print(f"  - {problem}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("FEATURE AUDIT: PASS")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
