import logging

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

try:
    from .config import DEBUG_FRAME_PATH, configure_logging
except ImportError:
    from config import DEBUG_FRAME_PATH, configure_logging

logger = logging.getLogger(__name__)

device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

mtcnn = None
if MTCNN is not None:
    mtcnn = MTCNN(
        image_size=224,
        margin=20,
        post_process=True,
        device=device
    )


def get_aligned_face(frame_bgr):
    """
    Detect and align a face.

    Parameters:
        frame_bgr : OpenCV frame

    Returns:
        torch.Tensor or None
    """

    if cv2 is None or mtcnn is None:
        raise ModuleNotFoundError("OpenCV and facenet-pytorch are required to detect faces.")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mtcnn(frame_rgb)


def main():
    configure_logging()
    frame = cv2.imread(str(DEBUG_FRAME_PATH))

    if frame is None:
        logger.error("Could not read debug frame: %s", DEBUG_FRAME_PATH)
        return

    face = get_aligned_face(frame)

    if face is None:
        logger.info("No face detected.")
    else:
        logger.info("Face tensor shape: %s", tuple(face.shape))


if __name__ == "__main__":
    main()