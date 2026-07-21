import librosa
import matplotlib.pyplot as plt

audio_path = "temp_audio/WhatsApp Video 2026-07-11 at 6.27.47 PM.wav"

waveform, sr = librosa.load(audio_path, sr=16000)

mel = librosa.feature.melspectrogram(
    y=waveform,
    sr=sr,
    n_mels=80
)

print(mel.shape)