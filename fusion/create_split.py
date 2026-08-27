"""
Build the persistent, group-aware train/validation/test split and save it to
fusion/data_split.json. Run this ONCE (or whenever the feature set changes);
every training/evaluation script then loads the same file instead of
re-splitting on its own.

Usage:
    python fusion/create_split.py \\
        --visual-root visual/data/features_aligned \\
        --audio-root audio/data/features \\
        --semantic-root C:\\Deepfake_Features\\semantic_features \\
        --blink-root visual/data/blink_features \\
        --lipsync-root visual/data/lipsync_features

    # Re-check an existing split without regenerating it:
    python fusion/create_split.py --verify-only

Sample set: a relative path is included only if it exists under ALL FIVE
roots (visual/audio/semantic/blink/lipsync) - the same criterion
EnhancedFusionDataset uses. That guarantees anything in the split is usable
by the 3-modal fusion model, the 5-modal enhanced model, and every
single-modality classifier alike.

Grouping and target ratios: see split_utils.py's module docstring for the
identity-token grouping method. Group assignment to train/validation/test
uses a deterministic (seed=42) shuffle followed by a greedy
largest-remaining-deficit bin-packing pass targeting 70/15/15 by SAMPLE
count. Because groups have uneven sizes, the achieved ratios will be close
to but not exactly 70/15/15 - the achieved numbers are computed and printed
(and saved in the metadata block) rather than assumed.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import (  # noqa: E402
    DEFAULT_SPLIT_PATH,
    build_groups,
    group_size_diagnostics,
    load_split,
    to_key,
    verify_split,
)

SEED = 42
TARGET_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def collect_aligned_samples(visual_root, audio_root, semantic_root, blink_root, lipsync_root):
    visual_root = Path(visual_root)
    audio_root = Path(audio_root)
    semantic_root = Path(semantic_root)
    blink_root = Path(blink_root)
    lipsync_root = Path(lipsync_root)

    samples = []  # list of (relative_path_key, label)
    skipped_incomplete = 0

    for visual_file in sorted(visual_root.rglob("*.npy")):
        relative_path = visual_file.relative_to(visual_root)
        key = to_key(relative_path)

        if not (audio_root / relative_path).exists():
            skipped_incomplete += 1
            continue
        if not (semantic_root / relative_path).exists():
            skipped_incomplete += 1
            continue
        if not (blink_root / relative_path).exists():
            skipped_incomplete += 1
            continue
        if not (lipsync_root / relative_path).exists():
            skipped_incomplete += 1
            continue

        first_folder = relative_path.parts[0]
        if first_folder.startswith("RealVideo"):
            label = 0
        elif first_folder.startswith("FakeVideo"):
            label = 1
        else:
            continue

        samples.append((key, label))

    return samples, skipped_incomplete


def assign_groups_to_splits(samples):
    """samples: list of (relative_path_key, label). Returns dict key -> split_name."""
    keys = [k for k, _ in samples]
    labels = dict(samples)
    group_of = build_groups(keys)

    group_members = {}
    for k in keys:
        group_members.setdefault(group_of[k], []).append(k)

    group_list = sorted(group_members.keys())  # deterministic order before shuffling
    rng = random.Random(SEED)
    rng.shuffle(group_list)
    # Bin-pack largest groups first for a tighter fit to the target ratios;
    # the seeded shuffle above still controls tie-breaking order.
    group_list.sort(key=lambda g: len(group_members[g]), reverse=True)

    total = len(keys)
    target_counts = {name: ratio * total for name, ratio in TARGET_RATIOS.items()}
    current_counts = {name: 0 for name in TARGET_RATIOS}

    key_to_split = {}
    for g in group_list:
        members = group_members[g]
        # Assign this whole group to whichever split is furthest below its
        # target count (largest remaining deficit) - standard greedy
        # bin-packing for balanced partitioning under a grouping constraint.
        deficits = {name: target_counts[name] - current_counts[name] for name in TARGET_RATIOS}
        chosen = max(deficits, key=deficits.get)
        for k in members:
            key_to_split[k] = chosen
        current_counts[chosen] += len(members)

    return key_to_split, group_of, group_members


def build_split_document(samples, key_to_split, group_of, meta_extra):
    labels = dict(samples)
    counts = {name: {"total": 0, "real": 0, "fake": 0} for name in TARGET_RATIOS}
    for key, split_name in key_to_split.items():
        counts[split_name]["total"] += 1
        counts[split_name]["real" if labels[key] == 0 else "fake"] += 1

    total = len(samples)
    achieved_ratios = {name: (counts[name]["total"] / total if total else 0.0) for name in TARGET_RATIOS}

    # Diagnostics only - does NOT change which samples land in which group
    # or split. Purpose: detect if the union-find identity grouping (see
    # split_utils.py's module docstring on the "hub node" risk from fake
    # clips' embedded second identity token) accidentally merged a large
    # fraction of the dataset into one giant group, which would then have
    # to be assigned to a single split wholesale.
    diagnostics = group_size_diagnostics(group_of)

    doc = {
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": SEED,
            "target_ratios": TARGET_RATIOS,
            "achieved_ratios": achieved_ratios,
            "total_samples": total,
            "total_groups": len(set(group_of.values())),
            "group_method": "union-find over id<digits> tokens found in folder names + filename; "
                            "samples with no id token are singleton groups keyed by their own path",
            "group_size_diagnostics": diagnostics,
            "modalities_required_for_inclusion": ["visual", "audio", "semantic", "blink", "lipsync"],
            "counts": counts,
            **meta_extra,
        },
        "samples": {
            key: {"split": key_to_split[key], "label": labels[key], "group": group_of[key]}
            for key in key_to_split
        },
    }
    return doc


def main():
    parser = argparse.ArgumentParser(description="Create the persistent group-aware train/validation/test split.")
    parser.add_argument("--visual-root", default="visual/data/features_aligned")
    parser.add_argument("--audio-root", default="audio/data/features")
    parser.add_argument("--semantic-root", default=None,
                         help="Defaults to $DFD_SEMANTIC_ROOT if set, else semantic/data/features.")
    parser.add_argument("--blink-root", default="visual/data/blink_features")
    parser.add_argument("--lipsync-root", default="visual/data/lipsync_features")
    parser.add_argument("--output", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--verify-only", action="store_true",
                         help="Skip regeneration; just load --output and print the verification report.")
    args = parser.parse_args()

    if args.verify_only:
        split_data = load_split(args.output)
        report = verify_split(split_data)
        _print_report(report, split_data)
        sys.exit(0 if report["ok"] else 1)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from env_defaults import DEFAULT_SEMANTIC_ROOT  # noqa: E402

    semantic_root = args.semantic_root or DEFAULT_SEMANTIC_ROOT

    print("Scanning feature roots:")
    print(f"  visual   : {args.visual_root}")
    print(f"  audio    : {args.audio_root}")
    print(f"  semantic : {semantic_root}")
    print(f"  blink    : {args.blink_root}")
    print(f"  lipsync  : {args.lipsync_root}")

    samples, skipped_incomplete = collect_aligned_samples(
        args.visual_root, args.audio_root, semantic_root, args.blink_root, args.lipsync_root
    )
    print(f"\nAligned samples (present in all 5 streams): {len(samples)}")
    print(f"Skipped (missing from at least one stream):  {skipped_incomplete}")

    if not samples:
        print("No aligned samples found - check the five root paths above.")
        sys.exit(1)

    key_to_split, group_of, group_members = assign_groups_to_splits(samples)
    doc = build_split_document(
        samples, key_to_split, group_of,
        meta_extra={
            "visual_root": str(args.visual_root),
            "audio_root": str(args.audio_root),
            "semantic_root": str(semantic_root),
            "blink_root": str(args.blink_root),
            "lipsync_root": str(args.lipsync_root),
        },
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"\nSaved split to {output_path}")

    report = verify_split(doc)
    _print_report(report, doc)
    sys.exit(0 if report["ok"] else 1)


def _print_report(report, split_data):
    meta = split_data.get("metadata", {})
    print("\n" + "=" * 60)
    print("SPLIT VERIFICATION")
    print("=" * 60)
    print(f"Total samples : {report['total_samples']}")
    print(f"Total groups  : {meta.get('total_groups', 'n/a')}")

    diagnostics = meta.get("group_size_diagnostics")
    if diagnostics:
        print(f"  Largest group : {diagnostics['largest_group_size']} sample(s) "
              f"({diagnostics['largest_group_share'] * 100:.1f}% of dataset), key={diagnostics['largest_group_key']}")
        print(f"  Median group  : {diagnostics['median_group_size']} sample(s)")
        print(f"  Singleton groups (no id token found): {diagnostics['singleton_group_count']} of "
              f"{diagnostics['total_groups']}")
        if diagnostics["warning"]:
            # Flagged for a human to look at, not acted on automatically -
            # the split itself is never changed because of this, and this
            # warning never fails the split (see the exit-code logic at
            # the bottom of this function, which is independent of it).
            print(f"\n  MANUAL REVIEW WARNING: {diagnostics['warning']}")

    for name in ("train", "validation", "test"):
        dist = report["label_distribution"].get(name, {"real": 0, "fake": 0, "total": 0, "fake_percentage": 0.0})
        overall_pct = (dist["total"] / report["total_samples"] * 100) if report["total_samples"] else 0.0
        print(f"  {name:<10} total={dist['total']:<6} ({overall_pct:5.1f}% of all samples)   "
              f"real={dist['real']:<6} fake={dist['fake']:<6} fake%={dist['fake_percentage']:.1f}%")

    if report["flat_overlaps"]:
        print("\nFAIL - overlapping keys between splits:")
        for pair, n in report["flat_overlaps"].items():
            print(f"  {pair}: {n} overlapping samples")
    else:
        print("\nOK - no sample appears in more than one split.")

    if report["group_violations"]:
        print(f"\nFAIL - {len(report['group_violations'])} identity group(s) span multiple splits:")
        for v in report["group_violations"][:10]:
            print(f"  group {v['group']} -> {v['splits']}")
        if len(report["group_violations"]) > 10:
            print(f"  ... and {len(report['group_violations']) - 10} more")
    else:
        print("OK - every identity group is entirely within one split (no group-level leakage).")

    if report["class_violations"]:
        print(f"\nFAIL - {len(report['class_violations'])} split(s) contain only one class:")
        for v in report["class_violations"]:
            print(f"  {v['split']}: real={v['real']} fake={v['fake']} - cannot train/evaluate meaningfully on this")
    else:
        print("OK - every non-empty split contains both real and fake samples.")

    file_existence = report["file_existence"]
    if not file_existence.get("checked"):
        print(f"\nSKIPPED - file-existence re-check: {file_existence.get('note', 'not checked')}")
    elif file_existence["missing"]:
        print(f"\nFAIL - {len(file_existence['missing'])} sample(s) missing from at least one feature root "
              f"(showing up to {len(file_existence['missing'])}):")
        for m in file_existence["missing"][:10]:
            print(f"  {m['relative_path']}: missing from {m['missing_from']}")
        if len(file_existence["missing"]) > 10:
            print(f"  ... and more (list truncated)")
    else:
        print("OK - every sample still exists under all five recorded feature roots.")

    print("=" * 60)
    if not report["ok"]:
        print("SPLIT VERIFICATION FAILED - do not proceed to training/evaluation until this is resolved.")
        print("=" * 60)


if __name__ == "__main__":
    main()
