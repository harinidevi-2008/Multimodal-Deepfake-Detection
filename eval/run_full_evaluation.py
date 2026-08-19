"""
Runs every evaluation script in sequence and prints one consolidated
report: visual/audio/semantic classifiers, original fusion, enhanced
fusion, and the rule-based blink/lip-sync scores.

This is an orchestrator, not a reimplementation - it calls each
existing script (classifiers/evaluate_classifier.py,
fusion/evaluate_fusion.py, fusion/evaluate_enhanced_fusion.py,
fusion/evaluate_blink_lipsync.py) as a subprocess and streams their
output, so there's exactly one place each metric is actually computed.
Skips a section gracefully (prints a note, moves on) if the weights or
feature roots for that section don't exist yet - useful while some
pieces are trained and others aren't.

Edit the paths dict below to match where things live on your machine,
or override via CLI flags.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(label, cmd, cwd):
    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[skipped/failed] {label}: {result.stderr.strip().splitlines()[-1] if result.stderr else 'unknown error'}")


def main():
    parser = argparse.ArgumentParser(description="Run the full backend evaluation suite.")
    parser.add_argument("--visual-root", default=str(REPO_ROOT / "visual" / "data" / "features_aligned"))
    parser.add_argument("--audio-root", default=str(REPO_ROOT / "audio" / "data" / "features"))
    parser.add_argument("--semantic-root", default=str(REPO_ROOT / "semantic" / "data" / "features"))
    parser.add_argument("--blink-root", default=str(REPO_ROOT / "visual" / "data" / "blink_features"))
    parser.add_argument("--lipsync-root", default=str(REPO_ROOT / "visual" / "data" / "lipsync_features"))
    parser.add_argument("--classifiers-dir", default=str(REPO_ROOT / "classifiers"))
    parser.add_argument("--fusion-weights", default=str(REPO_ROOT / "fusion" / "best_fusion_model.pt"))
    parser.add_argument(
        "--enhanced-fusion-weights", default=str(REPO_ROOT / "fusion" / "best_enhanced_fusion_model.pt")
    )
    args = parser.parse_args()

    py = sys.executable

    modality_specs = [
        ("visual", args.visual_root, 1280),
        ("audio", args.audio_root, 768),
        ("semantic", args.semantic_root, 384),
    ]

    for modality, root, dim in modality_specs:
        weights = Path(args.classifiers_dir) / f"best_{modality}_classifier.pt"
        run(
            f"{modality.upper()} classifier",
            [
                py, str(REPO_ROOT / "classifiers" / "evaluate_classifier.py"),
                "--modality", modality, "--feature-root", root,
                "--input-dim", str(dim), "--weights", str(weights),
            ],
            cwd=REPO_ROOT,
        )

    run(
        "FUSION (original, 3-modality)",
        [
            py, str(REPO_ROOT / "fusion" / "evaluate_fusion.py"),
            "--visual-root", args.visual_root, "--audio-root", args.audio_root,
            "--semantic-root", args.semantic_root, "--weights", args.fusion_weights,
        ],
        cwd=REPO_ROOT / "fusion",
    )

    run(
        "ENHANCED FUSION (5-modality)",
        [
            py, str(REPO_ROOT / "fusion" / "evaluate_enhanced_fusion.py"),
            "--visual-root", args.visual_root, "--audio-root", args.audio_root,
            "--semantic-root", args.semantic_root, "--blink-root", args.blink_root,
            "--lipsync-root", args.lipsync_root, "--weights", args.enhanced_fusion_weights,
        ],
        cwd=REPO_ROOT / "fusion",
    )

    run(
        "BLINK / LIP-SYNC (rule-based, not a trained classifier)",
        [
            py, str(REPO_ROOT / "fusion" / "evaluate_blink_lipsync.py"),
            "--blink-root", args.blink_root, "--lipsync-root", args.lipsync_root,
        ],
        cwd=REPO_ROOT / "fusion",
    )

    print(f"\n{'=' * 60}")
    print("Full evaluation run complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
