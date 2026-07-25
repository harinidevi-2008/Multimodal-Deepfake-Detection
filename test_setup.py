import logging

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

try:
    from visual.src.config import configure_logging
except ImportError:
    from config import configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()
    logger.info("PyTorch Version : %s", torch.__version__)
    logger.info("CUDA Available  : %s", torch.cuda.is_available())
    logger.info("OpenCV Version  : %s", cv2.__version__)
    logger.info("NumPy Version   : %s", np.__version__)

    mtcnn = MTCNN()
    logger.info("✅ MTCNN Loaded Successfully!")


if __name__ == "__main__":
    main()