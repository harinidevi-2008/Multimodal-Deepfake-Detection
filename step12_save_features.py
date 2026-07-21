import librosa
import torch
import numpy as np

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model

# Load Audio
audio_path = "temp_audio/WhatsApp Video 2026-07-11 at 6.27.47 PM.wav"

waveform, sample_rate = librosa.load(audio_path, sr=16000)

# Load Processor and Model
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

# Convert audio into model input
inputs = processor(
    waveform,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

# Extract features
with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.last_hidden_state

print("Original Shape :", hidden_states.shape)

# Mean Pooling
feature_vector = hidden_states.mean(dim=1)

print("Final Feature Shape :", feature_vector.shape)

# Save Features
np.save(
    "audio_features/audio_feature.npy",
    feature_vector.squeeze().numpy()
)

print("Feature Vector Saved Successfully")