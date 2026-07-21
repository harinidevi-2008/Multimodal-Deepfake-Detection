import librosa

audio_path = "temp_audio/WhatsApp Video 2026-07-11 at 6.27.47 PM.wav"

waveform, sample_rate = librosa.load(audio_path, sr=16000)

print("Sample Rate:", sample_rate)
print("Waveform Shape:", waveform.shape)
print("First 20 Values:")
print(waveform[:20])