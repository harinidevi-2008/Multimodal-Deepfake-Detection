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

effnet = None
if torch is not None and nn is not None and models is not None:
    effnet = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )
    effnet.classifier = nn.Identity()
    effnet.eval()
    effnet.to(device)

    for param in effnet.parameters():
        param.requires_grad = False


def get_feature_vector(face_tensor):
    if effnet is None or torch is None:
        raise ModuleNotFoundError("PyTorch and torchvision are required to extract features.")

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