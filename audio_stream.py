import os
import numpy as np
import librosa
import torch

from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from projection import AudioProjection

# =====================================================
# CONFIGURATION
# =====================================================

VIDEO_FOLDER = "sample_videos"
AUDIO_FOLDER = "temp_audio"
FEATURE_FOLDER = "audio_features"

SAMPLE_RATE = 16000

os.makedirs(AUDIO_FOLDER, exist_ok=True)
os.makedirs(FEATURE_FOLDER, exist_ok=True)

# =====================================================
# LOAD WAV2VEC2 PROCESSOR
# =====================================================

print("\nLoading Wav2Vec2 Processor...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("Processor Loaded Successfully")

# =====================================================
# LOAD WAV2VEC2 MODEL
# =====================================================

print("\nLoading Wav2Vec2 Model...")

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

model.eval()

print("Model Loaded Successfully\n")

# =====================================================
# LOAD AUDIO PROJECTION MODEL
# =====================================================

print("Loading Projection Model...")

projection_model = AudioProjection()

projection_model.eval()

print("Projection Model Loaded Successfully\n")

# =====================================================
# PROCESS EACH VIDEO
# =====================================================

video_files = [
    f for f in os.listdir(VIDEO_FOLDER)
    if f.lower().endswith((".mp4", ".avi", ".mov"))
]

if not video_files:
    print("No videos found inside sample_videos/")
    exit()

for video in video_files:

    print("=" * 60)
    print("Processing:", video)

    video_path = os.path.join(VIDEO_FOLDER, video)

    filename = os.path.splitext(video)[0]

    audio_path = os.path.join(
        AUDIO_FOLDER,
        filename + ".wav"
    )

    feature_path = os.path.join(
        FEATURE_FOLDER,
        filename + ".npy"
    )

    # =================================================
    # STEP 1 : Extract Audio
    # =================================================

    if not os.path.exists(audio_path):

        print("Extracting Audio...")

        clip = VideoFileClip(video_path)

        clip.audio.write_audiofile(
            audio_path,
            fps=SAMPLE_RATE,
            verbose=False,
            logger=None
        )

        clip.close()

    else:
        print("Audio already exists.")

    # =================================================
    # STEP 2 : Load Audio
    # =================================================

    print("Loading Audio...")

    waveform, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    print("Sample Rate :", sr)
    print("Waveform Shape :", waveform.shape)

    # =================================================
    # STEP 3 : Generate Mel Spectrogram
    # =================================================

    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=80
    )

    print("Mel Spectrogram Shape :", mel.shape)

    # =================================================
    # STEP 4 : Prepare Input
    # =================================================

    inputs = processor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    # =================================================
    # STEP 5 : Extract Features
    # =================================================

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state

    print("Hidden State Shape :", hidden_states.shape)

    # =================================================
    # STEP 6 : Mean Pooling
    # =================================================

    feature_vector = hidden_states.mean(dim=1)

    print("768-D Feature Shape :", feature_vector.shape)

    # =================================================
    # STEP 7 : Projection (768 -> 256)
    # =================================================

    with torch.no_grad():
        projected_feature = projection_model(feature_vector)

    print("256-D Feature Shape :", projected_feature.shape)

    # =================================================
    # STEP 8 : Save Projected Feature
    # =================================================

    np.save(
        feature_path,
        projected_feature.squeeze().cpu().numpy()
    )

    print("Saved Projected Feature :", feature_path)

print("\n=============================================")
print("All videos processed successfully.")
print("=============================================")