import librosa
import torch

from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model

# -------------------------------
# Load Audio
# -------------------------------

audio_path = "temp_audio/WhatsApp Video 2026-07-11 at 6.27.47 PM.wav"

waveform, sample_rate = librosa.load(
    audio_path,
    sr=16000
)

print("Audio Loaded")
print("Sample Rate:", sample_rate)
print("Waveform Shape:", waveform.shape)

# -------------------------------
# Load Processor & Model
# -------------------------------

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("Model Loaded")

# -------------------------------
# Convert Audio to Model Input
# -------------------------------

inputs = processor(
    waveform,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True
)

print("Processor Finished")

# -------------------------------
# Extract Features
# -------------------------------

with torch.no_grad():
    outputs = model(**inputs)

print("Feature Extraction Completed")

# -------------------------------
# Hidden States
# -------------------------------

hidden_states = outputs.last_hidden_state

print("Hidden State Shape:", hidden_states.shape)