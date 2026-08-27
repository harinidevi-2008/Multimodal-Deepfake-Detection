import logging
from pathlib import Path

try:
    from .config import OUTPUT_FOLDER, VIDEO_FOLDER, configure_logging
    from .process_video import process_video
except ImportError:
    from config import OUTPUT_FOLDER, VIDEO_FOLDER, configure_logging
    from process_video import process_video

logger = logging.getLogger(__name__)


def extract(video_path):
    video_path = Path(video_path)
    output_file = video_path.stem + ".npy"
    output_path = OUTPUT_FOLDER / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = process_video(str(video_path), str(output_path))
    return result


def main():
    configure_logging()
    sample_video = VIDEO_FOLDER / "real_harini_001.mp4"
    logger.info("Extracting features for %s", sample_video)
    extract(sample_video)


if __name__ == "__main__":
    main()
