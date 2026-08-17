import whisper
import torch
import os


print("Loading Whisper base model...")
model = whisper.load_model("base")
print("Whisper loaded successfully")
print("Total parameters:", sum(p.numel() for p in model.parameters()))
print()


def transcribe_video(video_path):
    if not os.path.exists(video_path):
        print("File not found:", video_path)
        return {
            "text": "",
            "confidence": -2.0,
            "language": "unknown",
            "reliable": False,
            "segments": []
        }

    print("Transcribing:", video_path)

    result = model.transcribe(
        str(video_path),
        language=None,
        task="transcribe",
        verbose=False,
        fp16=False
    )

    transcript = result["text"].strip()
    segments = result.get("segments", [])

    if segments:
        confidence = sum(s["avg_logprob"] for s in segments) / len(segments)
    else:
        confidence = -2.0

    if len(transcript.split()) < 3:
        confidence -= 0.5

    return {
        "text": transcript,
        "confidence": confidence,
        "language": result.get("language", "unknown"),
        "reliable": confidence > -1.0,
        "segments": segments
    }


def show_result(result):
    print()
    print("=" * 55)
    print("TRANSCRIPTION RESULT")
    print("=" * 55)
    print("Transcript :", result["text"])
    print("Language   :", result["language"])
    print("Confidence :", round(result["confidence"], 3))
    print("Reliable   :", result["reliable"])
    print("Segments   :", len(result["segments"]), "chunks")
    print()

    conf = result["confidence"]
    if conf >= -0.3:
        quality = "Excellent — very clear speech"
    elif conf >= -0.6:
        quality = "Good — slight background noise"
    elif conf >= -1.0:
        quality = "Okay — some noise, still usable"
    elif conf >= -1.5:
        quality = "Poor — transcript may have errors"
    else:
        quality = "Failed — do not use this transcript"

    print("Quality    :", quality)
    print()

    if result["segments"]:
        print("Segment breakdown:")
        print("  Start     End    Conf   Text")
        print("  ------  ------  ------  --------------------")
        for seg in result["segments"]:
            start = seg.get("start", 0)
            end   = seg.get("end", 0)
            conf  = seg.get("avg_logprob", -2.0)
            text  = seg.get("text", "").strip()
            print(f"  {start:>5.1f}s  {end:>5.1f}s  {conf:>6.2f}  {text}")

    print("=" * 55)


def test_without_video():
    print("Testing Whisper internal components...")
    print()

    # Test 1: Check model loaded correctly
    total = sum(p.numel() for p in model.parameters())
    print("Test 1 — Model parameters:", total)
    assert total > 70000000, "Model seems too small"
    print("         PASSED")
    print()

    # Test 2: Check audio to mel spectrogram conversion
    print("Test 2 — Audio processing pipeline...")
    fake_audio = torch.zeros(48000, dtype=torch.float32)
    mel = whisper.log_mel_spectrogram(fake_audio)
    print("         Audio shape:", fake_audio.shape)
    print("         Mel shape  :", mel.shape)
    print("         PASSED")
    print()

    # Test 3: Check model structure counts
    print("Test 3 — Model structure...")
    enc = sum(p.numel() for p in model.encoder.parameters())
    dec = sum(p.numel() for p in model.decoder.parameters())
    tot = sum(p.numel() for p in model.parameters())
    print("         Encoder parameters:", enc)
    print("         Decoder parameters:", dec)
    print("         Total parameters  :", tot)
    assert enc > 0, "Encoder missing"
    assert dec > 0, "Decoder missing"
    print("         PASSED")
    print()

    # Test 4: Check all parameters are accessible
    print("Test 4 — Parameter check...")
    all_params = list(model.parameters())
    print("         Total parameter tensors:", len(all_params))
    assert len(all_params) > 0, "No parameters found"
    print("         PASSED")
    print()

    print("All internal tests passed.")
    print("Whisper is ready to transcribe real video files.")


test_without_video()
print()

video_path = "sample1.mp4"

if os.path.exists(video_path):
    print("Testing with real video:", video_path)
    result = transcribe_video(video_path)
    show_result(result)
else:
    print("No video file found.")
    print("To test with a real video:")
    print("  1. Copy any MP4 video into this folder")
    print("  2. Rename it to sample.mp4")
    print("  3. Run this file again")
    print()
    print("Whisper internal tests already passed above.")
