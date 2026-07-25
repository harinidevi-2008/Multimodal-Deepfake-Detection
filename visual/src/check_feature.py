import logging

import numpy as np

try:
    from .config import OUTPUT_FOLDER, configure_logging
except ImportError:
    from config import OUTPUT_FOLDER, configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()
    feature_path = OUTPUT_FOLDER / "real_harini_001.npy"
    feature = np.load(feature_path)

    logger.info("Shape : %s", feature.shape)
    logger.info("First 10 values:")
    logger.info("%s", feature[:10])
    logger.info("Data type: %s", feature.dtype)


if __name__ == "__main__":
    main()