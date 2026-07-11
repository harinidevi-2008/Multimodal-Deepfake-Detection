import torch
import torch.nn as nn
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

effnet = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
)

effnet.classifier = nn.Identity()

effnet.eval()

effnet.to(device)

for param in effnet.parameters():
    param.requires_grad = False


def get_feature_vector(face_tensor):

    with torch.no_grad():

        face_tensor = face_tensor.unsqueeze(0).to(device)

        feature = effnet(face_tensor)

    return feature.squeeze(0).cpu().numpy()


if __name__ == "__main__":

    import cv2
    from face_detector import get_aligned_face

    frame = cv2.imread("visual/debug/extracted_frames/frame_000.jpg")

    face = get_aligned_face(frame)

    feature = get_feature_vector(face)

    print(feature.shape)