"""
Train-only normalization statistics for the compact blink and lip-sync
feature vectors.

Why only these two modalities: the visual/audio/semantic embeddings
already pass through a Linear -> LayerNorm projection inside
FusionModel / EnhancedFusionModel, which standardizes each projected
dimension itself - there was no reviewed reason to also z-score the
raw 1280/768/384-d embeddings before that. The blink vector (4-d:
a count, a rate-per-minute, a duration in seconds, a 0-1 score) and
the lip-sync vector (2-d, both already 0-1 scores but on a different
scale of variation than the blink count/rate/duration mix) combine
values with mismatched units in one short vector with no learned
normalization step before the shared projection layer - these are the
genuine candidates for explicit normalization.

Statistics are fit on the TRAIN split only (fusion/data_split.json,
produced by fusion/create_split.py) and must then be applied
IDENTICALLY at train/eval/inference time - fitting on anything other
than train, or applying different statistics at inference than were
used during training, would defeat the point.

Usage:
    python fusion/feature_normalization.py \\
        --blink-root visual/data/blink_features \\
        --lipsync-root visual/data/lipsync_features \\
        --split-path fusion/data_split.json \\
        --output fusion/feature_normalization.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_utils import DEFAULT_SPLIT_PATH, load_split, split_of  # noqa: E402

DEFAULT_NORMALIZATION_PATH = Path(__file__).resolve().parent / "feature_normalization.json"

BLINK_DIM_NAMES = ["blink_count", "blink_rate_per_min", "average_blink_duration_sec", "blink_irregularity_score"]
LIPSYNC_DIM_NAMES = ["sync_score", "mismatch_score"]


def _collect_train_vectors(feature_root, split_data):
    feature_root = Path(feature_root)
    vectors = []
    for npy_file in sorted(feature_root.rglob("*.npy")):
        relative_path = npy_file.relative_to(feature_root)
        if split_of(relative_path, split_data) != "train":
            continue
        vectors.append(np.load(npy_file).astype(np.float64))
    return np.stack(vectors) if vectors else np.zeros((0,))


def fit_stats(vectors, dim_names, eps=1e-8):
    """
    vectors: 2-D array, shape (n_samples, len(dim_names)).

    Returns {'dims', 'mean', 'std', 'n_samples', 'flat_dims'}. A
    near-zero std (a dimension that is effectively constant across the
    training split) is clamped to 1.0 rather than eps, so that
    dimension is only mean-centered instead of being blown up by
    dividing by a near-zero number; 'flat_dims' names which dimensions
    that happened to, so it can be surfaced rather than silently
    hidden.
    """
    if vectors.shape[0] == 0:
        raise ValueError("No training-split vectors found to fit normalization on.")

    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    flat_dims = [dim_names[i] for i, s in enumerate(std) if s < eps]
    std_safe = np.where(std < eps, 1.0, std)

    return {
        "dims": dim_names,
        "mean": mean.tolist(),
        "std": std_safe.tolist(),
        "n_samples": int(vectors.shape[0]),
        "flat_dims": flat_dims,
    }


def apply_normalization(vector, stats):
    """
    vector: 1-D np.ndarray. stats: {'mean': [...], 'std': [...]} (one
    modality's block from a loaded feature_normalization.json, or from
    fit_stats() directly). Returns a new float32 array - does not
    mutate the input.
    """
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((np.asarray(vector, dtype=np.float32) - mean) / std).astype(np.float32)


def load_normalization(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No normalization file at {path}. Run `python fusion/feature_normalization.py` first, "
            "or pass --no-normalization to train/evaluate without it."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------
# Checkpoint <-> normalization sidecar metadata
# ---------------------------------------------------------------------
#
# A checkpoint's raw state_dict has no memory of whether it was trained
# with blink/lip-sync normalization applied. Until now, train/eval/
# inference each independently decided whether to normalize based only
# on their own CLI flags (--no-normalization) and whatever
# feature_normalization.json happened to be sitting on disk - so
# training a checkpoint WITH normalization and then evaluating or
# running inference on it WITHOUT (or vice versa) produced numbers
# that were wrong, but not visibly wrong. This sidecar file closes
# that gap: it is written next to a checkpoint at save time recording
# what was actually used to train it, and can be checked at eval/
# inference time against what is about to be applied.

def normalization_meta_path(checkpoint_path):
    """Sidecar metadata path for a checkpoint (<checkpoint>.normalization_meta.json)."""
    return Path(str(checkpoint_path) + ".normalization_meta.json")


def save_normalization_meta(checkpoint_path, normalization_applied, normalization_path=None):
    """
    Writes the sidecar recording whether `checkpoint_path` was trained
    with blink/lip-sync normalization applied, and with which file.
    Call this every time a checkpoint is (re)written, so the sidecar
    never drifts out of sync with what is actually on disk.
    """
    meta = {
        "normalization_applied": bool(normalization_applied),
        "normalization_path": (str(normalization_path) if normalization_applied and normalization_path else None),
    }
    meta_path = normalization_meta_path(checkpoint_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def load_normalization_meta(checkpoint_path):
    """Returns the sidecar dict, or None if it doesn't exist (e.g. a checkpoint saved before this check existed)."""
    meta_path = normalization_meta_path(checkpoint_path)
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_normalization_consistency(checkpoint_path, normalization_applied_now):
    """
    Compares what a checkpoint was actually trained with (per its
    sidecar metadata, if any) against what is about to be applied at
    evaluation/inference time. Returns a human-readable warning string
    if there is a detected mismatch or the checkpoint predates this
    check (unknown - can't be verified), or None if consistent.

    Deliberately a warning, not a hard failure: --no-normalization is
    an intentional, explicit experimental knob (see train_enhanced_fusion.py),
    and a checkpoint saved before this pass existed has no sidecar at
    all - neither case should block a run outright, but both must stop
    being silent.
    """
    meta = load_normalization_meta(checkpoint_path)
    if meta is None:
        return (
            f"[normalization check] No sidecar metadata found for checkpoint '{checkpoint_path}' "
            f"(expected '{normalization_meta_path(checkpoint_path).name}' beside it). This checkpoint "
            "was likely trained before this check existed, or its sidecar was deleted. Train/eval "
            "normalization consistency cannot be verified automatically here - confirm manually that "
            "--no-normalization matches how this checkpoint was actually trained."
        )
    trained_with_normalization = meta.get("normalization_applied")
    if trained_with_normalization != bool(normalization_applied_now):
        return (
            f"[normalization MISMATCH] Checkpoint '{checkpoint_path}' was trained with "
            f"normalization_applied={trained_with_normalization}, but this run is about to apply "
            f"normalization_applied={bool(normalization_applied_now)}. Results will likely be silently "
            "wrong - pass --no-normalization (or the matching --normalization-path) so this run matches "
            "how the checkpoint was actually trained."
        )
    return None


def main():
    parser = argparse.ArgumentParser(description="Fit train-only blink/lip-sync normalization statistics.")
    parser.add_argument("--blink-root", required=True)
    parser.add_argument("--lipsync-root", required=True)
    parser.add_argument("--split-path", default=str(DEFAULT_SPLIT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_NORMALIZATION_PATH))
    args = parser.parse_args()

    split_data = load_split(args.split_path)

    blink_vectors = _collect_train_vectors(args.blink_root, split_data)
    lipsync_vectors = _collect_train_vectors(args.lipsync_root, split_data)

    blink_stats = fit_stats(blink_vectors, BLINK_DIM_NAMES)
    lipsync_stats = fit_stats(lipsync_vectors, LIPSYNC_DIM_NAMES)

    doc = {
        "metadata": {
            "fit_on_split": "train",
            "split_path": str(args.split_path),
            "blink_root": str(args.blink_root),
            "lipsync_root": str(args.lipsync_root),
        },
        "blink": blink_stats,
        "lipsync": lipsync_stats,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"Fit blink normalization from {blink_stats['n_samples']} train samples.")
    for name, m, s in zip(BLINK_DIM_NAMES, blink_stats["mean"], blink_stats["std"]):
        print(f"  {name:<30} mean={m:.4f}  std={s:.4f}")
    if blink_stats["flat_dims"]:
        print(f"  NOTE: near-constant in train split (left at std=1.0, not rescaled): {blink_stats['flat_dims']}")

    print(f"\nFit lip-sync normalization from {lipsync_stats['n_samples']} train samples.")
    for name, m, s in zip(LIPSYNC_DIM_NAMES, lipsync_stats["mean"], lipsync_stats["std"]):
        print(f"  {name:<30} mean={m:.4f}  std={s:.4f}")
    if lipsync_stats["flat_dims"]:
        print(f"  NOTE: near-constant in train split (left at std=1.0, not rescaled): {lipsync_stats['flat_dims']}")

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
