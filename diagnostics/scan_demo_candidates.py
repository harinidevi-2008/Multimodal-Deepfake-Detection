"""Find numerically valid demo candidates from the persisted test split.

This only reads existing feature files and checkpoints. It intentionally uses
the same five-stream attention model, modality order, train-fit normalization,
and calibrated threshold as inference/full_pipeline.py.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
FUSION_DIR = REPO_ROOT / "fusion"

import sys

sys.path.insert(0, str(FUSION_DIR))

from enhanced_fusion_dataset import EnhancedFusionDataset
from enhanced_fusion_model import EnhancedFusionModel
from feature_normalization import load_normalization


MODALITY_ROOTS = {
    "visual": REPO_ROOT / "visual" / "data" / "features_aligned",
    "audio": REPO_ROOT / "audio" / "data" / "features",
    "semantic": REPO_ROOT / "semantic" / "data" / "features",
    "blink": REPO_ROOT / "visual" / "data" / "blink_features",
    "lipsync": REPO_ROOT / "visual" / "data" / "lipsync_features",
}
SPLIT_PATH = FUSION_DIR / "data_split.json"
NORMALIZATION_PATH = FUSION_DIR / "feature_normalization.json"
WEIGHTS_PATH = FUSION_DIR / "best_enhanced_fusion_model_attention.pt"
THRESHOLDS_PATH = REPO_ROOT / "eval" / "results" / "thresholds.json"


def is_valid_feature(vector):
    return bool(vector.size and np.isfinite(vector).all() and np.max(np.abs(vector)) > 1e-8)


def main():
    split_doc = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    threshold = float(split_doc.get("threshold", 0.0))
    threshold_doc = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    threshold = float(threshold_doc["thresholds"]["enhanced_attention"]["threshold"])
    normalization = load_normalization(NORMALIZATION_PATH)

    dataset = EnhancedFusionDataset(
        visual_root=MODALITY_ROOTS["visual"],
        audio_root=MODALITY_ROOTS["audio"],
        semantic_root=MODALITY_ROOTS["semantic"],
        blink_root=MODALITY_ROOTS["blink"],
        lipsync_root=MODALITY_ROOTS["lipsync"],
        split_path=SPLIT_PATH,
        split_name="test",
        normalization_stats=normalization,
        return_reliability=True,
        cache_features=False,
    )
    relative_paths = [path.relative_to(MODALITY_ROOTS["visual"]) for path, *_ in dataset.samples]

    model = EnhancedFusionModel(fusion_architecture="attention")
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu"), strict=True)
    model.eval()

    records = []
    cursor = 0
    with torch.inference_mode():
        for visual, audio, semantic, blink, lipsync, labels, reliability in DataLoader(
            dataset, batch_size=256, shuffle=False
        ):
            logits, _ = model(visual, audio, semantic, blink, lipsync, reliability=reliability)
            probabilities = torch.softmax(logits, dim=1)
            for index, label in enumerate(labels.tolist()):
                relative_path = relative_paths[cursor]
                cursor += 1
                raw_features = {
                    name: np.load(root / relative_path).astype(np.float32)
                    for name, root in MODALITY_ROOTS.items()
                }
                validity = {name: is_valid_feature(vector) for name, vector in raw_features.items()}
                fake_probability = float(probabilities[index, 1])
                records.append({
                    "sample_id": str(relative_path),
                    "raw_video_path": str(
                        REPO_ROOT / "datasets" / "FakeAVCeleb_v1.2" / relative_path.with_suffix(".mp4")
                    ),
                    "raw_video_exists": (REPO_ROOT / "datasets" / "FakeAVCeleb_v1.2" / relative_path.with_suffix(".mp4")).is_file(),
                    "ground_truth": int(label),
                    "real_probability": float(probabilities[index, 0]),
                    "fake_probability": fake_probability,
                    "prediction": "FAKE" if fake_probability >= threshold else "REAL",
                    "threshold": threshold,
                    "margin_from_threshold": abs(fake_probability - threshold),
                    "semantic_reliable": bool(reliability[index, 2].item()),
                    "all_five_modalities_valid": all(validity.values()),
                    "validity": validity,
                })

    eligible = [
        record for record in records
        if record["semantic_reliable"] and record["all_five_modalities_valid"]
    ]
    strong_fake = sorted(
        (record for record in eligible if record["ground_truth"] == 1 and record["prediction"] == "FAKE"),
        key=lambda record: record["fake_probability"], reverse=True,
    )[:10]
    strong_real = sorted(
        (record for record in eligible if record["ground_truth"] == 0 and record["prediction"] == "REAL"),
        key=lambda record: record["fake_probability"],
    )[:10]
    print(json.dumps({
        "checkpoint": str(WEIGHTS_PATH),
        "modality_order": ["visual", "audio", "semantic", "blink", "lipsync"],
        "threshold": threshold,
        "test_samples": len(records),
        "eligible_samples": len(eligible),
        "strong_fake": strong_fake,
        "strong_real": strong_real,
    }, indent=2))


if __name__ == "__main__":
    main()
