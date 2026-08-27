"""
Runs every evaluation script in sequence and prints one consolidated
report: visual/audio/semantic classifiers, original fusion, enhanced
fusion, and the rule-based blink/lip-sync scores.

This is an orchestrator, not a reimplementation - it calls each
existing script (classifiers/evaluate_classifier.py,
fusion/evaluate_fusion.py, fusion/evaluate_enhanced_fusion.py,
fusion/evaluate_blink_lipsync.py) as a subprocess and streams their
output, so there's exactly one place each metric is actually computed.

Honest status per component (see run()'s docstring): "NOT TRAINED"
(no checkpoint exists - never even attempted, distinct from a real
failure), "NO DATA" (the sub-script itself reported zero aligned
samples for the requested split and exited non-zero for that reason -
evaluate_fusion.py / evaluate_enhanced_fusion.py / evaluate_classifier.py
all now sys.exit(1) on this instead of silently returning success),
"FAILED" (an actual crash/exception), or "SUCCESS". A final summary
table lists every component's status so nothing gets silently reported
as passing when it never ran.

Paths default from fusion/env_defaults.py (override via CLI flags or
the DFD_* environment variables) instead of being hardcoded here.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "fusion"))
from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_ROOT,
    DEFAULT_BLINK_ROOT,
    DEFAULT_ENHANCED_FUSION_WEIGHTS,
    DEFAULT_FUSION_WEIGHTS,
    DEFAULT_LIPSYNC_ROOT,
    DEFAULT_SEMANTIC_ROOT,
    DEFAULT_VISUAL_ROOT,
)


def resolve_path(p):
    """
    Resolve a possibly-relative path argument against REPO_ROOT - NOT
    the current working directory, and NOT the subprocess's cwd. This
    fixes a real bug: the fusion/enhanced-fusion/blink-lipsync
    sub-scripts are launched with cwd=REPO_ROOT/"fusion" (so that their
    own relative-import machinery works), but every default path in
    fusion/env_defaults.py (e.g. "fusion/best_fusion_model.pt",
    "visual/data/features_aligned") is written relative to REPO_ROOT.
    Passed through unresolved, the subprocess would look for
    REPO_ROOT/fusion/fusion/best_fusion_model.pt instead of
    REPO_ROOT/fusion/best_fusion_model.pt - a checkpoint that exists
    would be silently reported as "NOT TRAINED". Resolving every
    path-like argument to an absolute path here, before any subprocess
    command is built, makes the result independent of both the
    subprocess's cwd and the directory this orchestrator itself was
    launched from. An already-absolute path (e.g. a user-supplied
    override) is returned unchanged.
    """
    if p is None:
        return None
    path = Path(p)
    return str(path if path.is_absolute() else (REPO_ROOT / path).resolve())


def run(label, cmd, cwd, weights_path=None):
    """
    Returns one of "SUCCESS", "FAILED", "NOT TRAINED", "NO DATA" - see
    module docstring for what each means.
    """
    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)

    if weights_path is not None and not Path(weights_path).exists():
        print(f"[not trained] {label}: no checkpoint at {weights_path} - skipping.")
        return "NOT TRAINED"

    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode == 0:
        return "SUCCESS"

    combined_output = (result.stdout or "") + (result.stderr or "")
    stderr_tail = result.stderr.strip().splitlines()[-1] if result.stderr else "unknown error"

    if "no samples found" in combined_output.lower() or "no aligned samples" in combined_output.lower():
        print(f"[no data] {label}: {stderr_tail}")
        return "NO DATA"

    print(f"[failed] {label}: {stderr_tail}")
    return "FAILED"


def main():
    parser = argparse.ArgumentParser(description="Run the full backend evaluation suite.")
    parser.add_argument("--visual-root", default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--semantic-root", default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--blink-root", default=DEFAULT_BLINK_ROOT)
    parser.add_argument("--lipsync-root", default=DEFAULT_LIPSYNC_ROOT)
    parser.add_argument("--classifiers-dir", default=str(REPO_ROOT / "classifiers"))
    parser.add_argument("--fusion-weights", default=DEFAULT_FUSION_WEIGHTS)
    parser.add_argument("--enhanced-fusion-weights", default=DEFAULT_ENHANCED_FUSION_WEIGHTS)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"],
                         help="Forwarded to every sub-script that supports --split. Defaults to 'test' - "
                              "the held-out set. 'all' is NOT a final metric, debugging only.")
    parser.add_argument("--split-path", default=None,
                         help="Path to data_split.json. Omit to use each sub-script's own default "
                              "(fusion/data_split.json).")
    parser.add_argument("--normalization-path", default=None,
                         help="Forwarded to evaluate_enhanced_fusion.py. Omit to use its own default "
                              "(fusion/feature_normalization.json if present, else raw features).")
    args = parser.parse_args()

    # Resolve every filesystem-path argument against REPO_ROOT before
    # building any subprocess command - see resolve_path()'s docstring.
    # This must happen before the checkpoint-existence checks below too,
    # so "NOT TRAINED" reflects reality regardless of cwd.
    args.visual_root = resolve_path(args.visual_root)
    args.audio_root = resolve_path(args.audio_root)
    args.semantic_root = resolve_path(args.semantic_root)
    args.blink_root = resolve_path(args.blink_root)
    args.lipsync_root = resolve_path(args.lipsync_root)
    args.classifiers_dir = resolve_path(args.classifiers_dir)
    args.fusion_weights = resolve_path(args.fusion_weights)
    args.enhanced_fusion_weights = resolve_path(args.enhanced_fusion_weights)
    args.split_path = resolve_path(args.split_path)
    args.normalization_path = resolve_path(args.normalization_path)

    if args.split == "all":
        print("WARNING: --split all evaluates on every sample regardless of train/val/test membership. "
              "This number is NOT a valid held-out test metric - debugging only.\n")

    py = sys.executable
    statuses = {}

    split_flags = ["--split", args.split]
    if args.split_path:
        split_flags += ["--split-path", args.split_path]

    modality_specs = [
        ("visual", args.visual_root, 1280),
        ("audio", args.audio_root, 768),
        ("semantic", args.semantic_root, 384),
    ]

    for modality, root, dim in modality_specs:
        weights = Path(args.classifiers_dir) / f"best_{modality}_classifier.pt"
        statuses[f"{modality} classifier"] = run(
            f"{modality.upper()} classifier",
            [
                py, str(REPO_ROOT / "classifiers" / "evaluate_classifier.py"),
                "--modality", modality, "--feature-root", root,
                "--input-dim", str(dim), "--weights", str(weights),
            ] + split_flags,
            cwd=REPO_ROOT,
            weights_path=weights,
        )

    statuses["fusion (3-modal baseline)"] = run(
        "FUSION (original, 3-modality baseline)",
        [
            py, str(REPO_ROOT / "fusion" / "evaluate_fusion.py"),
            "--visual-root", args.visual_root, "--audio-root", args.audio_root,
            "--semantic-root", args.semantic_root, "--weights", args.fusion_weights,
        ] + split_flags,
        cwd=REPO_ROOT / "fusion",
        weights_path=args.fusion_weights,
    )

    enhanced_fusion_cmd = [
        py, str(REPO_ROOT / "fusion" / "evaluate_enhanced_fusion.py"),
        "--visual-root", args.visual_root, "--audio-root", args.audio_root,
        "--semantic-root", args.semantic_root, "--blink-root", args.blink_root,
        "--lipsync-root", args.lipsync_root, "--weights", args.enhanced_fusion_weights,
    ] + split_flags
    if args.normalization_path:
        enhanced_fusion_cmd += ["--normalization-path", args.normalization_path]

    statuses["enhanced fusion (5-modal)"] = run(
        "ENHANCED FUSION (5-modality)",
        enhanced_fusion_cmd,
        cwd=REPO_ROOT / "fusion",
        weights_path=args.enhanced_fusion_weights,
    )

    statuses["blink/lip-sync (rule-based)"] = run(
        "BLINK / LIP-SYNC (rule-based, not a trained classifier - fixed thresholds, split-aware since "
        "Priority 4 of the hardening pass)",
        [
            py, str(REPO_ROOT / "fusion" / "evaluate_blink_lipsync.py"),
            "--blink-root", args.blink_root, "--lipsync-root", args.lipsync_root,
        ] + split_flags,
        cwd=REPO_ROOT / "fusion",
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for name, status in statuses.items():
        print(f"  {name:<32} {status}")
    print("=" * 60)

    any_failed = any(s == "FAILED" for s in statuses.values())
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
