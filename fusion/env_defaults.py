"""
Centralized, environment-variable-overridable default paths for the
fusion pipeline.

Why this exists: several scripts (train_enhanced_fusion.py,
fusion_inference.py, eval/run_full_evaluation.py) previously hardcoded
r"C:\\Deepfake_Features\\semantic_features" as the semantic feature
root's default value, because on the development machine that is where
semantic features actually live (semantic/data/features was not used).
That is fine as a personal default but breaks for anyone else who
clones the repo. Every script that needs one of these paths should now
import its default from here (or accept an explicit CLI flag), and set
the corresponding environment variable to override it for their own
machine instead of editing source files.

None of these are guesses at "what a correct default should be" beyond
the repo-relative convention every other stream already uses
(<stream>/data/<features>) - only the semantic root differs from that
convention by design (see above), which is exactly why it is the one
most worth making overridable.
"""

import os

DEFAULT_VISUAL_ROOT = os.environ.get("DFD_VISUAL_ROOT", "visual/data/features_aligned")
DEFAULT_AUDIO_ROOT = os.environ.get("DFD_AUDIO_ROOT", "audio/data/features")
DEFAULT_SEMANTIC_ROOT = os.environ.get("DFD_SEMANTIC_ROOT", "semantic/data/features")
DEFAULT_BLINK_ROOT = os.environ.get("DFD_BLINK_ROOT", "visual/data/blink_features")
DEFAULT_LIPSYNC_ROOT = os.environ.get("DFD_LIPSYNC_ROOT", "visual/data/lipsync_features")

DEFAULT_FUSION_WEIGHTS = os.environ.get("DFD_FUSION_WEIGHTS", "fusion/best_fusion_model.pt")
DEFAULT_ENHANCED_FUSION_WEIGHTS = os.environ.get(
    "DFD_ENHANCED_FUSION_WEIGHTS", "fusion/best_enhanced_fusion_model.pt"
)

DEFAULT_VISUAL_CLASSIFIER_WEIGHTS = os.environ.get(
    "DFD_VISUAL_CLASSIFIER_WEIGHTS", "classifiers/best_visual_classifier.pt"
)
DEFAULT_AUDIO_CLASSIFIER_WEIGHTS = os.environ.get(
    "DFD_AUDIO_CLASSIFIER_WEIGHTS", "classifiers/best_audio_classifier.pt"
)
DEFAULT_SEMANTIC_CLASSIFIER_WEIGHTS = os.environ.get(
    "DFD_SEMANTIC_CLASSIFIER_WEIGHTS", "classifiers/best_semantic_classifier.pt"
)
