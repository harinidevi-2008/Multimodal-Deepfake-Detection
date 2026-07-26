import logging

from transformers import Wav2Vec2Model, Wav2Vec2Processor

try:
    from .config import MODEL_NAME, configure_logging
except ImportError:
    from config import MODEL_NAME, configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()
    logger.info("Loading Processor...")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    logger.info("Processor Loaded")

    logger.info("Loading Model...")
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    logger.info("Model Loaded Successfully")


if __name__ == "__main__":
    main()
