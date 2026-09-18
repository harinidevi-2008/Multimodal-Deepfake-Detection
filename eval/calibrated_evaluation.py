import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "classifiers"))
sys.path.insert(0, str(REPO_ROOT / "fusion"))

from single_stream_classifier import SingleStreamClassifier  # noqa: E402
from single_stream_dataset import SingleStreamDataset  # noqa: E402
from fusion_dataset import FusionDataset  # noqa: E402
from fusion_model import FusionModel  # noqa: E402
from enhanced_fusion_dataset import EnhancedFusionDataset  # noqa: E402
from enhanced_fusion_model import EnhancedFusionModel  # noqa: E402
from env_defaults import (  # noqa: E402
    DEFAULT_AUDIO_ROOT,
    DEFAULT_BLINK_ROOT,
    DEFAULT_LIPSYNC_ROOT,
    DEFAULT_SEMANTIC_ROOT,
    DEFAULT_VISUAL_ROOT,
)
from feature_normalization import DEFAULT_NORMALIZATION_PATH, load_normalization  # noqa: E402
from split_utils import DEFAULT_SPLIT_PATH  # noqa: E402


THRESHOLD_CRITERION = (
    "Maximize validation balanced accuracy over unique validation fake-probability scores; "
    "break ties by validation F1, then by threshold closest to 0.5. Test labels are never used."
)


def architecture_for_checkpoint(weights_path, fallback="attention"):
    meta_path = Path(str(weights_path) + ".model_meta.json")
    if not meta_path.exists():
        return fallback
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f).get("fusion_architecture", fallback)


def _candidate_thresholds(probs):
    values = np.unique(np.asarray(probs, dtype=np.float64))
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], values)))


def calibrate_threshold(labels, probs):
    best = None
    for threshold in _candidate_thresholds(probs):
        preds = (probs >= threshold).astype(int)
        bal_acc = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        key = (bal_acc, f1, -abs(float(threshold) - 0.5))
        if best is None or key > best["key"]:
            best = {
                "threshold": float(threshold),
                "validation_balanced_accuracy": float(bal_acc),
                "validation_f1": float(f1),
                "key": key,
            }
    best.pop("key")
    return best


def compute_metrics(labels, probs, threshold):
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    fake_recall = tp / (tp + fn) if (tp + fn) else float("nan")

    try:
        roc_auc = roc_auc_score(labels, probs)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(labels, probs)
    except ValueError:
        pr_auc = float("nan")

    return {
        "n_samples": int(labels.size),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "specificity": float(specificity),
        "real_recall": float(specificity),
        "fake_recall": float(fake_recall),
        "confusion_matrix": cm.tolist(),
    }


@torch.no_grad()
def probs_for_loader(model, loader, device, kind):
    labels_out = []
    probs_out = []
    for batch in loader:
        if kind == "single":
            features, labels = batch
            logits = model(features.to(device))
        elif kind == "fusion3":
            visual, audio, semantic, labels = batch
            logits, _ = model(visual.to(device), audio.to(device), semantic.to(device))
        elif kind == "enhanced":
            visual, audio, semantic, blink, lipsync, labels, reliability = batch
            logits, _ = model(
                visual.to(device),
                audio.to(device),
                semantic.to(device),
                blink.to(device),
                lipsync.to(device),
                reliability=reliability.to(device),
            )
        else:
            raise ValueError(f"unknown loader kind: {kind}")
        labels_out.extend(labels.tolist())
        probs_out.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return np.asarray(labels_out), np.asarray(probs_out)


def eval_single(name, feature_root, input_dim, weights, split_path, batch_size, device):
    model = SingleStreamClassifier(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    def run(split):
        dataset = SingleStreamDataset(feature_root, split_path=split_path, split_name=split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return probs_for_loader(model, loader, device, "single")

    return _calibrated_result(name, weights, run)


def eval_fusion3(name, roots, weights, split_path, batch_size, device):
    model = FusionModel().to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()

    def run(split):
        dataset = FusionDataset(
            roots["visual"], roots["audio"], roots["semantic"], split_path=split_path, split_name=split
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return probs_for_loader(model, loader, device, "fusion3")

    return _calibrated_result(name, weights, run)


def eval_enhanced(name, roots, weights, split_path, normalization, batch_size, device):
    architecture = architecture_for_checkpoint(weights)
    model = EnhancedFusionModel(fusion_architecture=architecture).to(device)
    load_result = model.load_state_dict(torch.load(weights, map_location=device), strict=False)
    model.eval()

    def run(split):
        dataset = EnhancedFusionDataset(
            roots["visual"],
            roots["audio"],
            roots["semantic"],
            roots["blink"],
            roots["lipsync"],
            split_path=split_path,
            split_name=split,
            normalization_stats=normalization,
            cache_features=True,
            return_reliability=True,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return probs_for_loader(model, loader, device, "enhanced")

    result = _calibrated_result(name, weights, run)
    result["architecture"] = architecture
    result["strict_false_missing_keys"] = list(load_result.missing_keys)
    result["strict_false_unexpected_keys"] = list(load_result.unexpected_keys)
    return result


def _calibrated_result(name, weights, run_split):
    val_labels, val_probs = run_split("validation")
    threshold_info = calibrate_threshold(val_labels, val_probs)
    test_labels, test_probs = run_split("test")
    metrics = compute_metrics(test_labels, test_probs, threshold_info["threshold"])
    return {
        "name": name,
        "weights": str(weights),
        "threshold": threshold_info,
        "test": metrics,
    }


def print_result(result):
    test = result["test"]
    print(f"\n=== {result['name']} ===")
    if "architecture" in result:
        print(f"Architecture: {result['architecture']}")
    print(f"Validation threshold: {test['threshold']:.6f}")
    print(f"Test total: {test['n_samples']}  real: {test['n_real']}  fake: {test['n_fake']}")
    print(f"Accuracy:          {test['accuracy']:.4f}")
    print(f"Precision:         {test['precision']:.4f}")
    print(f"Recall:            {test['recall']:.4f}")
    print(f"F1:                {test['f1']:.4f}")
    print(f"ROC-AUC:           {test['roc_auc']:.4f}")
    print(f"PR-AUC:            {test['pr_auc']:.4f}")
    print(f"Balanced accuracy: {test['balanced_accuracy']:.4f}")
    print(f"Specificity:       {test['specificity']:.4f}")
    print(f"Real recall:       {test['real_recall']:.4f}")
    print(f"Fake recall:       {test['fake_recall']:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order=[real, fake]):")
    print(np.asarray(test["confusion_matrix"]))


def main():
    parser = argparse.ArgumentParser(description="Validation-calibrated evaluation for learned checkpoints.")
    parser.add_argument("--visual-root", default=DEFAULT_VISUAL_ROOT)
    parser.add_argument("--audio-root", default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--semantic-root", default=DEFAULT_SEMANTIC_ROOT)
    parser.add_argument("--blink-root", default=DEFAULT_BLINK_ROOT)
    parser.add_argument("--lipsync-root", default=DEFAULT_LIPSYNC_ROOT)
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--normalization-path", default=str(DEFAULT_NORMALIZATION_PATH))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-thresholds", default=str(REPO_ROOT / "eval" / "results" / "thresholds.json"))
    parser.add_argument("--output-results", default=str(REPO_ROOT / "eval" / "results" / "calibrated_results.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    roots = {
        "visual": args.visual_root,
        "audio": args.audio_root,
        "semantic": args.semantic_root,
        "blink": args.blink_root,
        "lipsync": args.lipsync_root,
    }
    normalization = load_normalization(args.normalization_path)

    jobs = [
        ("visual", lambda: eval_single("Visual", args.visual_root, 1280, REPO_ROOT / "classifiers" / "best_visual_classifier.pt", args.split_path, args.batch_size, device)),
        ("audio", lambda: eval_single("Audio", args.audio_root, 768, REPO_ROOT / "classifiers" / "best_audio_classifier.pt", args.split_path, args.batch_size, device)),
        ("semantic", lambda: eval_single("Semantic", args.semantic_root, 384, REPO_ROOT / "classifiers" / "best_semantic_classifier.pt", args.split_path, args.batch_size, device)),
        ("fusion_3_stream", lambda: eval_fusion3("3-stream Fusion", roots, REPO_ROOT / "fusion" / "best_fusion_model.pt", args.split_path, args.batch_size, device)),
        ("enhanced_base", lambda: eval_enhanced("5-stream Enhanced Fusion", roots, REPO_ROOT / "fusion" / "best_enhanced_fusion_model.pt", args.split_path, normalization, args.batch_size, device)),
        ("enhanced_attention", lambda: eval_enhanced("5-stream Enhanced Fusion Attention", roots, REPO_ROOT / "fusion" / "best_enhanced_fusion_model_attention.pt", args.split_path, normalization, args.batch_size, device)),
        ("enhanced_gated", lambda: eval_enhanced("5-stream Enhanced Fusion Gated", roots, REPO_ROOT / "fusion" / "best_enhanced_fusion_model_gated.pt", args.split_path, normalization, args.batch_size, device)),
    ]

    results = {}
    for key, fn in jobs:
        result = fn()
        results[key] = result
        print_result(result)

    thresholds = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "criterion": THRESHOLD_CRITERION,
        "split_path": str(args.split_path),
        "thresholds": {
            key: {
                "threshold": value["threshold"]["threshold"],
                "validation_balanced_accuracy": value["threshold"]["validation_balanced_accuracy"],
                "validation_f1": value["threshold"]["validation_f1"],
                "weights": value["weights"],
            }
            for key, value in results.items()
        },
    }
    result_doc = {
        "generated_utc": thresholds["generated_utc"],
        "threshold_criterion": THRESHOLD_CRITERION,
        "results": results,
    }

    for output_path, doc in [(args.output_thresholds, thresholds), (args.output_results, result_doc)]:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
