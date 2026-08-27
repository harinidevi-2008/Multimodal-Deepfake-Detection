"""
Consolidated experiment report: calls each evaluation module's
importable evaluate(...) function directly (no subprocess + stdout
parsing) and writes one combined eval/results.json (machine-readable)
and eval/results.txt (human-readable) covering every component that
has trained weights available.

This is a report generator, not a new evaluation implementation - the
actual metric computation still happens in exactly one place per
component:
  A. Learned single-stream classifiers (visual/audio/semantic) -
     classifiers/evaluate_classifier.py's evaluate()
  B. 3-modal learned fusion (visual+audio+semantic) -
     fusion/evaluate_fusion.py's evaluate()
  C. 5-modal learned fusion (visual+audio+semantic+blink+lipsync) -
     fusion/evaluate_enhanced_fusion.py's evaluate()
  D. Rule-based behavioral signals (blink, lip-sync) -
     fusion/evaluate_blink_lipsync.py's evaluate_blink()/evaluate_lipsync()
     - NOT learned probabilities, reported with their own fixed
     threshold and clearly labeled "rule-based" throughout.

Usage:
    python eval/generate_report.py --split test
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "fusion"))
sys.path.insert(0, str(REPO_ROOT / "classifiers"))

from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_ROOT,
    DEFAULT_BLINK_ROOT,
    DEFAULT_ENHANCED_FUSION_WEIGHTS,
    DEFAULT_FUSION_WEIGHTS,
    DEFAULT_LIPSYNC_ROOT,
    DEFAULT_SEMANTIC_ROOT,
    DEFAULT_VISUAL_ROOT,
)
from feature_normalization import DEFAULT_NORMALIZATION_PATH  # noqa: E402
from evaluate_fusion import evaluate as evaluate_fusion  # noqa: E402
from evaluate_enhanced_fusion import evaluate as evaluate_enhanced_fusion  # noqa: E402
from evaluate_classifier import evaluate as evaluate_classifier  # noqa: E402
from evaluate_blink_lipsync import evaluate_blink, evaluate_lipsync  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate a consolidated eval/results.json + results.txt.")
    parser.add_argument("--visual-root", default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--semantic-root", default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--blink-root", default=DEFAULT_BLINK_ROOT)
    parser.add_argument("--lipsync-root", default=DEFAULT_LIPSYNC_ROOT)
    parser.add_argument("--classifiers-dir", default=str(REPO_ROOT / "classifiers"))
    parser.add_argument("--fusion-weights", default=DEFAULT_FUSION_WEIGHTS)
    parser.add_argument("--enhanced-fusion-weights", default=DEFAULT_ENHANCED_FUSION_WEIGHTS)
    parser.add_argument("--normalization-path", default=str(DEFAULT_NORMALIZATION_PATH),
                         help="Forwarded to evaluate_enhanced_fusion(). Pass --no-normalization to skip it "
                              "even if the file exists.")
    parser.add_argument("--no-normalization", action="store_true")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test", "all"])
    parser.add_argument("--split-path", default=None)
    parser.add_argument("--output-json", default=str(REPO_ROOT / "eval" / "results.json"))
    parser.add_argument("--output-txt", default=str(REPO_ROOT / "eval" / "results.txt"))
    args = parser.parse_args()

    split_kwargs = {"split": args.split}
    if args.split_path:
        split_kwargs["split_path"] = args.split_path

    normalization_path = None if args.no_normalization else args.normalization_path

    components = {}

    # Every evaluate_*() call below is wrapped in try/except: an
    # unexpected exception must become an explicit "FAILED" component
    # with the error text recorded, never silently reported as
    # "SUCCESS" (which would need a result) or "NO DATA" (which means
    # the sub-function itself cleanly determined there were zero
    # samples and returned None - a different, non-exceptional case). A
    # final report must never look complete when a component actually
    # crashed.

    # --- A. Learned single-stream classifiers ---
    for modality, root, dim in [
        ("visual", args.visual_root, 1280),
        ("audio", args.audio_root, 768),
        ("semantic", args.semantic_root, 384),
    ]:
        weights = Path(args.classifiers_dir) / f"best_{modality}_classifier.pt"
        if not weights.exists():
            components[f"{modality}_classifier"] = {"status": "NOT TRAINED", "weights_path": str(weights),
                                                      "category": "A_single_stream_classifier"}
            continue
        try:
            result = evaluate_classifier(modality, root, dim, str(weights), **split_kwargs)
        except Exception as exc:  # noqa: BLE001 - must surface as FAILED, not crash the whole report
            components[f"{modality}_classifier"] = {"status": "FAILED", "category": "A_single_stream_classifier",
                                                      "error": str(exc)}
            continue
        if result is None:
            components[f"{modality}_classifier"] = {"status": "NO DATA", "category": "A_single_stream_classifier"}
        else:
            result.pop("probs", None)  # not JSON-serializable, and not needed in the report
            components[f"{modality}_classifier"] = {"status": "SUCCESS", "category": "A_single_stream_classifier",
                                                      **result}

    # --- B. 3-modal learned fusion (baseline) ---
    if not Path(args.fusion_weights).exists():
        components["fusion_3modal"] = {"status": "NOT TRAINED", "weights_path": args.fusion_weights,
                                        "category": "B_3modal_fusion"}
    else:
        try:
            result = evaluate_fusion(args.visual_root, args.audio_root, args.semantic_root,
                                      args.fusion_weights, **split_kwargs)
            components["fusion_3modal"] = (
                {"status": "SUCCESS", "category": "B_3modal_fusion", **result} if result
                else {"status": "NO DATA", "category": "B_3modal_fusion"}
            )
        except Exception as exc:  # noqa: BLE001 - must surface as FAILED, not crash the whole report
            components["fusion_3modal"] = {"status": "FAILED", "category": "B_3modal_fusion", "error": str(exc)}

    # --- C. 5-modal learned fusion ---
    if not Path(args.enhanced_fusion_weights).exists():
        components["fusion_5modal"] = {"status": "NOT TRAINED", "weights_path": args.enhanced_fusion_weights,
                                        "category": "C_5modal_fusion"}
    else:
        try:
            result = evaluate_enhanced_fusion(
                args.visual_root, args.audio_root, args.semantic_root, args.blink_root, args.lipsync_root,
                args.enhanced_fusion_weights, normalization_path=normalization_path, **split_kwargs,
            )
            components["fusion_5modal"] = (
                {"status": "SUCCESS", "category": "C_5modal_fusion", **result} if result
                else {"status": "NO DATA", "category": "C_5modal_fusion"}
            )
        except Exception as exc:  # noqa: BLE001 - must surface as FAILED, not crash the whole report
            components["fusion_5modal"] = {"status": "FAILED", "category": "C_5modal_fusion", "error": str(exc)}

    # --- D. Rule-based behavioral signals (NOT learned probabilities) ---
    try:
        blink_result = evaluate_blink(args.blink_root, **split_kwargs)
        components["blink_rule_based"] = (
            {"status": "SUCCESS", "category": "D_rule_based", **blink_result} if blink_result
            else {"status": "NO DATA", "category": "D_rule_based"}
        )
    except Exception as exc:  # noqa: BLE001 - must surface as FAILED, not crash the whole report
        components["blink_rule_based"] = {"status": "FAILED", "category": "D_rule_based", "error": str(exc)}

    try:
        lipsync_result = evaluate_lipsync(args.lipsync_root, **split_kwargs)
        components["lipsync_rule_based"] = (
            {"status": "SUCCESS", "category": "D_rule_based", **lipsync_result} if lipsync_result
            else {"status": "NO DATA", "category": "D_rule_based"}
        )
    except Exception as exc:  # noqa: BLE001 - must surface as FAILED, not crash the whole report
        components["lipsync_rule_based"] = {"status": "FAILED", "category": "D_rule_based", "error": str(exc)}

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": args.split,
        "categories": {
            "A_single_stream_classifier": "Learned single-stream classifiers (visual/audio/semantic)",
            "B_3modal_fusion": "3-modal learned fusion (visual+audio+semantic)",
            "C_5modal_fusion": "5-modal learned fusion (visual+audio+semantic+blink+lipsync)",
            "D_rule_based": "Rule-based behavioral signals (blink, lip-sync) - NOT learned probabilities",
        },
        "components": components,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    lines = [f"Evaluation report - split='{args.split}' - generated {doc['generated_utc']}", "=" * 60]
    for name, comp in components.items():
        lines.append(f"\n{name} [{comp.get('category', '?')}]: {comp['status']}")
        if comp["status"] == "SUCCESS" and comp["category"] in ("A_single_stream_classifier", "B_3modal_fusion", "C_5modal_fusion"):
            lines.append(f"  samples: {comp['n_samples']} (real={comp['n_real']}, fake={comp['n_fake']})")
            norm_note = ""
            if "normalization_applied" in comp:
                norm_note = f" normalization_applied={comp['normalization_applied']}"
            lines.append(
                f"  accuracy={comp['accuracy']:.4f} precision={comp['precision']:.4f} "
                f"recall={comp['recall']:.4f} f1={comp['f1']:.4f} roc_auc={comp['roc_auc']:.4f}{norm_note}"
            )
        elif comp["status"] == "SUCCESS" and comp["category"] == "D_rule_based":
            lines.append(f"  RULE-BASED - samples: {comp['n_samples']} (real={comp['n_real']}, fake={comp['n_fake']}), "
                         f"threshold={comp['threshold']}")
            lines.append(
                f"  accuracy={comp['accuracy']:.4f} precision={comp['precision']:.4f} "
                f"recall={comp['recall']:.4f} f1={comp['f1']:.4f} roc_auc={comp['roc_auc']:.4f}"
            )
        elif comp["status"] == "NOT TRAINED":
            lines.append(f"  no checkpoint at {comp.get('weights_path')}")
        elif comp["status"] == "FAILED":
            lines.append(f"  ERROR: {comp.get('error', 'unknown error')}")
    report_text = "\n".join(lines)

    with open(args.output_txt, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")

    print(report_text)
    print(f"\nSaved {args.output_json} and {args.output_txt}")

    any_failed = any(comp["status"] == "FAILED" for comp in components.values())
    if any_failed:
        print("\nAt least one component FAILED - see the ERROR line(s) above. This report is NOT complete.")
    sys.exit(1 if any_failed else 0)


if __name__ == "__main__":
    main()
