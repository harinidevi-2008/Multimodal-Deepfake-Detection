import cv2
import torch
from facenet_pytorch import MTCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

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

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    return mtcnn(frame_rgb)


if __name__ == "__main__":

    frame = cv2.imread("visual/debug/extracted_frames/frame_000.jpg")

    face = get_aligned_face(frame)

    if face is None:
        print("No face detected.")

    else:
        print(face.shape)