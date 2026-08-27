"""
Pytest-based synthetic regression test for EnhancedFusionDataset.

The previous version of this file hardcoded a machine-specific semantic
root (r"C:\\Deepfake_Features\\semantic_features") and executed
`dataset[0]` at module import time. On any machine without that exact
directory the dataset was empty and pytest crashed during COLLECTION
(IndexError: list index out of range) - not even a test failure, a
failure to even start the test suite. That is a test-design problem,
not a model problem: fixed here by building a tiny synthetic aligned
dataset under pytest's own tmp_path fixture (created fresh per test,
torn down automatically) and asserting real behavior against it. No
real FakeAVCeleb feature directories, no hardcoded paths, and no
dataset indexing happens at import time - every access is inside a
test function.
"""

from pathlib import Path

import numpy as np
import pytest

from enhanced_fusion_dataset import EnhancedFusionDataset

VISUAL_DIM = 1280
AUDIO_DIM = 768
SEMANTIC_DIM = 384
BLINK_DIM = 4
LIPSYNC_DIM = 2

ALL_DIMS = {
    "visual": VISUAL_DIM,
    "audio": AUDIO_DIM,
    "semantic": SEMANTIC_DIM,
    "blink": BLINK_DIM,
    "lipsync": LIPSYNC_DIM,
}


def _write_sample(roots, relative_path, dims):
    """Writes one random .npy per modality named in `dims` under its
    corresponding root, all at the same `relative_path` - i.e. an
    "aligned" sample. Passing a `dims` subset (missing one modality
    key) is how the "incomplete sample" test below is built."""
    for name, dim in dims.items():
        path = roots[name] / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.random.randn(dim).astype(np.float32))


@pytest.fixture
def synthetic_roots(tmp_path):
    """Five temporary feature-root directories, one per modality,
    created fresh for each test and cleaned up automatically by
    pytest's tmp_path fixture - never touches anything under the repo."""
    roots = {name: tmp_path / name for name in ALL_DIMS}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    return roots


def test_enhanced_fusion_dataset_loads_aligned_samples(synthetic_roots):
    # RealVideo-* / FakeVideo-* prefixes exercise the existing label
    # logic (first path component decides real=0 / fake=1).
    real_rel = Path("RealVideo-RealAudio/African/men/id00001/00001_0.npy")
    fake_rel = Path("FakeVideo-FakeAudio/African/men/id00002/00002_0_id00003_wavtolip.npy")

    _write_sample(synthetic_roots, real_rel, ALL_DIMS)
    _write_sample(synthetic_roots, fake_rel, ALL_DIMS)

    dataset = EnhancedFusionDataset(
        visual_root=synthetic_roots["visual"],
        audio_root=synthetic_roots["audio"],
        semantic_root=synthetic_roots["semantic"],
        blink_root=synthetic_roots["blink"],
        lipsync_root=synthetic_roots["lipsync"],
    )

    assert len(dataset) > 0
    assert len(dataset) == 2

    sample = dataset[0]
    assert len(sample) == 6

    visual, audio, semantic, blink, lipsync, label = sample
    assert tuple(visual.shape) == (VISUAL_DIM,)
    assert tuple(audio.shape) == (AUDIO_DIM,)
    assert tuple(semantic.shape) == (SEMANTIC_DIM,)
    assert tuple(blink.shape) == (BLINK_DIM,)
    assert tuple(lipsync.shape) == (LIPSYNC_DIM,)
    # Both fixture samples are real(0)/fake(1) by construction - either
    # order is valid depending on directory scan order, but the label
    # must be one of the two expected classes, never anything else.
    assert label.item() in (0, 1)


def test_enhanced_fusion_dataset_label_matches_folder_prefix(synthetic_roots):
    real_rel = Path("RealVideo-RealAudio/Caucasian/women/id00010/00010_0.npy")
    fake_rel = Path("FakeVideo-FakeAudio/Caucasian/women/id00011/00011_0_id00012_wavtolip.npy")

    _write_sample(synthetic_roots, real_rel, ALL_DIMS)
    _write_sample(synthetic_roots, fake_rel, ALL_DIMS)

    dataset = EnhancedFusionDataset(
        visual_root=synthetic_roots["visual"],
        audio_root=synthetic_roots["audio"],
        semantic_root=synthetic_roots["semantic"],
        blink_root=synthetic_roots["blink"],
        lipsync_root=synthetic_roots["lipsync"],
    )

    label_by_relative_path = {
        str(visual_file.relative_to(synthetic_roots["visual"])): label
        for visual_file, _, _, _, _, label in dataset.samples
    }
    assert label_by_relative_path[str(real_rel)] == 0
    assert label_by_relative_path[str(fake_rel)] == 1


def test_enhanced_fusion_dataset_excludes_sample_missing_one_modality(synthetic_roots):
    complete_rel = Path("RealVideo-RealAudio/African/men/id00020/00020_0.npy")
    incomplete_rel = Path("RealVideo-RealAudio/African/men/id00021/00021_0.npy")

    _write_sample(synthetic_roots, complete_rel, ALL_DIMS)
    # Deliberately omit the blink modality for this one sample - every
    # other modality file exists, so this exercises the "aligned across
    # all five roots" requirement specifically, not a totally-empty sample.
    dims_missing_blink = {name: dim for name, dim in ALL_DIMS.items() if name != "blink"}
    _write_sample(synthetic_roots, incomplete_rel, dims_missing_blink)

    dataset = EnhancedFusionDataset(
        visual_root=synthetic_roots["visual"],
        audio_root=synthetic_roots["audio"],
        semantic_root=synthetic_roots["semantic"],
        blink_root=synthetic_roots["blink"],
        lipsync_root=synthetic_roots["lipsync"],
    )

    relative_paths = {
        str(visual_file.relative_to(synthetic_roots["visual"]))
        for visual_file, _, _, _, _, _ in dataset.samples
    }
    assert str(complete_rel) in relative_paths
    assert str(incomplete_rel) not in relative_paths
    assert len(dataset) == 1
