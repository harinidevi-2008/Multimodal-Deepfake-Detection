import logging

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
except ImportError:
    torch = None
    nn = None
    models = None

try:
    from .config import DEBUG_FRAME_PATH, configure_logging
except ImportError:
    from config import DEBUG_FRAME_PATH, configure_logging

logger = logging.getLogger(__name__)

device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

# The EfficientNet-B0 backbone is built lazily (see _get_effnet() below),
# not at import time. Building it eagerly here used to mean that merely
# *importing* this module - which happens just by importing
# visual/src/__init__.py, which every other visual/src script and even
# unrelated pytest collection of visual/src/eye_blink or visual/src/lipsync
# scripts pulls in transitively - downloaded the pretrained ImageNet
# weights over the network. That broke test collection on an offline/
# restricted checkout and is the wrong behavior even with network access:
# nothing should download model weights as a side effect of an import.
_effnet = None


def _get_effnet():
    """Builds (and caches) the EfficientNet-B0 feature extractor on first
    use. Safe to call repeatedly - the network/download only happens once,
    on the first real call to get_feature_vector()."""
    global _effnet
    if _effnet is None:
        if torch is None or nn is None or models is None:
            raise ModuleNotFoundError("PyTorch and torchvision are required to extract features.")
        effnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        effnet.classifier = nn.Identity()
        effnet.eval()
        effnet.to(device)

        for param in effnet.parameters():
            param.requires_grad = False
        _effnet = effnet
    return _effnet


def get_feature_vector(face_tensor):
    if torch is None:
        raise ModuleNotFoundError("PyTorch and torchvision are required to extract features.")

    effnet = _get_effnet()

    with torch.no_grad():
        face_tensor = face_tensor.unsqueeze(0).to(device)
        feature = effnet(face_tensor)

    return feature.squeeze(0).cpu().numpy()


def main():
    configure_logging()

    import cv2

    try:
        from .face_detector import get_aligned_face
    except ImportError:
        from face_detector import get_aligned_face

    frame = cv2.imread(str(DEBUG_FRAME_PATH))

    if frame is None:
        logger.error("Could not read debug frame: %s", DEBUG_FRAME_PATH)
        return

    face = get_aligned_face(frame)

    if face is None:
        logger.info("No face detected.")
    else:
        feature = get_feature_vector(face)
        logger.info("Feature shape: %s", feature.shape)


if __name__ == "__main__":
    main()