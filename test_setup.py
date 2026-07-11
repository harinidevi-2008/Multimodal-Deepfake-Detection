import torch
import cv2
import numpy as np
from facenet_pytorch import MTCNN

print("PyTorch Version :", torch.__version__)
print("CUDA Available  :", torch.cuda.is_available())
print("OpenCV Version  :", cv2.__version__)
print("NumPy Version   :", np.__version__)

mtcnn = MTCNN()

print("✅ MTCNN Loaded Successfully!")