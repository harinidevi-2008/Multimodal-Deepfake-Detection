"""Regression checks for the binary class/probability contract at inference."""

import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
INFERENCE_DIR = Path(__file__).resolve().parent
for path in (REPO_ROOT, INFERENCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from full_pipeline import (  # noqa: E402
    FAKE_CLASS_INDEX,
    REAL_CLASS_INDEX,
    FeatureShapeError,
    IncompatibleCheckpointError,
    binary_class_probabilities,
    load_fusion_model,
    load_feature_vector,
    run_full_inference,
)
from fusion_model import FusionModel  # noqa: E402


class ProbabilityContractTests(unittest.TestCase):
    def test_binary_logits_map_index_zero_to_real_and_one_to_fake(self):
        probabilities = binary_class_probabilities(torch.tensor([[0.0, np.log(99.0)]]))

        self.assertEqual(REAL_CLASS_INDEX, 0)
        self.assertEqual(FAKE_CLASS_INDEX, 1)
        self.assertAlmostEqual(probabilities[0, REAL_CLASS_INDEX].item(), 0.01, places=6)
        self.assertAlmostEqual(probabilities[0, FAKE_CLASS_INDEX].item(), 0.99, places=6)
        self.assertAlmostEqual(probabilities.sum(dim=1).item(), 1.0, places=6)

    def test_binary_probability_conversion_rejects_invalid_logits(self):
        invalid_logits = (
            torch.tensor([0.0, 1.0]),
            torch.tensor([[0.0, 1.0, 2.0]]),
            torch.tensor([[float("nan"), 0.0]]),
        )
        for logits in invalid_logits:
            with self.subTest(logits=logits):
                with self.assertRaises(ValueError):
                    binary_class_probabilities(logits)

    def test_feature_loader_rejects_wrong_shape_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            bad_shape = tmp_path / "wrong_shape.npy"
            np.save(bad_shape, np.zeros((1279,), dtype=np.float32))
            with self.assertRaisesRegex(FeatureShapeError, r"expected \(1280,\)"):
                load_feature_vector(bad_shape, "visual")

            nonfinite = tmp_path / "nonfinite.npy"
            values = np.zeros((1280,), dtype=np.float32)
            values[17] = np.nan
            np.save(nonfinite, values)
            with self.assertRaisesRegex(FeatureShapeError, "NaN or infinite"):
                load_feature_vector(nonfinite, "visual")

    def test_feature_loader_preserves_valid_float32_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            source = np.arange(1280, dtype=np.float64)
            feature_path = Path(directory) / "visual.npy"
            np.save(feature_path, source)

            result = load_feature_vector(feature_path, "visual")

            self.assertEqual(result.shape, (1280,))
            self.assertEqual(result.dtype, np.float32)
            np.testing.assert_allclose(result, source.astype(np.float32))

    def test_fusion_model_accepts_only_three_streams_and_returns_logits(self):
        model = FusionModel(fusion_dim=16, num_heads=4).eval()
        visual, audio, semantic = torch.randn(2, 1280), torch.randn(2, 768), torch.randn(2, 384)
        logits, attention = model(visual, audio, semantic)
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(tuple(attention.shape), (2, 3, 3))
        probabilities = torch.softmax(logits, dim=1)
        self.assertTrue(torch.allclose(probabilities.sum(dim=1), torch.ones(2)))
        with self.assertRaises(TypeError):
            model(visual, audio, semantic, torch.zeros(2, 4), torch.zeros(2, 2))

    def test_supplementary_features_cannot_change_final_probability_or_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = {name: root / name for name in ("visual", "audio", "semantic", "blink", "lipsync")}
            vectors = {
                "visual": np.linspace(-0.1, 0.1, 1280, dtype=np.float32),
                "audio": np.linspace(-0.1, 0.1, 768, dtype=np.float32),
                "semantic": np.linspace(-0.1, 0.1, 384, dtype=np.float32),
            }
            for name, vector in vectors.items():
                roots[name].mkdir(parents=True)
                np.save(roots[name] / "sample.npy", vector)
            roots["blink"].mkdir(parents=True)
            roots["lipsync"].mkdir(parents=True)

            def infer(blink, lipsync):
                np.save(roots["blink"] / "sample.npy", np.array([2.0, 10.0, 0.2, blink], dtype=np.float32))
                np.save(roots["lipsync"] / "sample.npy", np.array([1.0 - lipsync, lipsync], dtype=np.float32))
                return run_full_inference(
                    "sample.npy",
                    **{f"{name}_root": str(path) for name, path in roots.items()},
                )

            normal = infer(0.0, 0.0)
            anomalous = infer(1.0, 1.0)

        self.assertEqual(normal["final_fake_probability"], anomalous["final_fake_probability"])
        self.assertEqual(normal["final_real_probability"], anomalous["final_real_probability"])
        self.assertEqual(normal["prediction"], anomalous["prediction"])
        self.assertEqual(normal["prediction"], "DEEPFAKE" if normal["final_fake_probability"] >= normal["decision_threshold"] else "REAL")

    def test_fusion_loader_rejects_missing_or_non_three_stream_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "fusion.pt"
            torch.save(FusionModel().state_dict(), checkpoint)
            with self.assertRaisesRegex(IncompatibleCheckpointError, "metadata is required"):
                load_fusion_model(checkpoint)

            metadata = {
                "model_class": "EnhancedFusionModel",
                "modalities": ["visual", "audio", "semantic", "blink", "lipsync"],
                "checkpoint_sha256": "irrelevant-for-wrong-architecture",
            }
            Path(str(checkpoint) + ".model_meta.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(IncompatibleCheckpointError, "not for the required three-stream"):
                load_fusion_model(checkpoint)


if __name__ == "__main__":
    unittest.main()
