"""Smoke tests for the production three-stream checkpoint and result UI contract."""

import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "fusion"))
from fusion_model import FusionModel  # noqa: E402


class ThreeStreamSmokeTests(unittest.TestCase):
    def test_shipped_checkpoint_is_metadata_verified_and_strict_loadable(self):
        checkpoint = ROOT / "fusion" / "best_fusion_model.pt"
        metadata = json.loads(Path(str(checkpoint) + ".model_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["model_class"], "FusionModel")
        self.assertEqual(metadata["modalities"], ["visual", "audio", "semantic"])
        model = FusionModel()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        model.eval()
        with torch.inference_mode():
            logits, attention = model(torch.zeros(1, 1280), torch.zeros(1, 768), torch.zeros(1, 384))
        self.assertEqual(tuple(logits.shape), (1, 2))
        self.assertEqual(tuple(attention.shape), (1, 3, 3))
        with self.assertRaises(TypeError):
            model(torch.zeros(1, 1280), torch.zeros(1, 768), torch.zeros(1, 384), torch.zeros(1, 4))

    def test_results_ui_has_three_learned_cards_and_supplementary_section_after_evidence(self):
        page = (ROOT / "frontend/src/screens/ResultsDashboard.jsx").read_text(encoding="utf-8")
        self.assertLess(page.index("<OverallResultPanel"), page.index("MODALITY_ORDER.map"))
        self.assertLess(page.index("MODALITY_ORDER.map"), page.index("<EvidenceViewer"))
        self.assertIn("grid-cols-3", page)
        self.assertLess(page.index("<EvidenceViewer"), page.index("Supplementary consistency checks"))
        modalities = (ROOT / "frontend/src/components/modalityMeta.js").read_text(encoding="utf-8")
        self.assertIn("MODALITY_ORDER = ['visual', 'audio', 'semantic']", modalities)
        self.assertIn("SUPPLEMENTARY_ORDER = ['eye_blink', 'lip_sync']", modalities)
        audio = (ROOT / "frontend/src/components/AudioTimeline.jsx").read_text(encoding="utf-8")
        self.assertIn("<audio", audio)
        self.assertIn("No timestamped suspicious audio interval", audio)


if __name__ == "__main__":
    unittest.main()
