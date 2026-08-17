import torch
import torch.nn as nn
import whisper
from sentence_transformers import SentenceTransformer, util
from pathlib import Path


class SemanticProjection(nn.Module):

    def __init__(self, input_dim=384, output_dim=256):
        super(SemanticProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        nn.init.xavier_uniform_(self.projection[0].weight)
        nn.init.zeros_(self.projection[0].bias)

    def forward(self, x):
        return self.projection(x)


class SemanticStream:

    def __init__(self, whisper_size="base", device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SemanticStream initialising on device: {self.device}")

        print(f"  Loading Whisper ({whisper_size})...")
        self.whisper_model = whisper.load_model(whisper_size)
        print(f"  Whisper ready — {sum(p.numel() for p in self.whisper_model.parameters()):,} params")

        print("  Loading Sentence-BERT (all-MiniLM-L6-v2)...")
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        for param in self.sbert_model.parameters():
            param.requires_grad = False

        frozen = sum(p.numel() for p in self.sbert_model.parameters() if not p.requires_grad)
        print(f"  Sentence-BERT ready — {frozen:,} params (all frozen)")

        self.projection = SemanticProjection(input_dim=384, output_dim=256).to(self.device)
        trainable = sum(p.numel() for p in self.projection.parameters() if p.requires_grad)
        print(f"  Projection ready — {trainable:,} trainable params")

        self._verify_setup()
        print("SemanticStream ready.\n")

    def _verify_setup(self):
        with torch.no_grad():
            dummy = torch.randn(384)
            out = self.projection(dummy)
        assert out.shape == torch.Size([256])
        assert out.dtype == torch.float32
        trainable = sum(p.numel() for p in self.sbert_model.parameters() if p.requires_grad)
        assert trainable == 0
        print("  Internal check passed.")

    def transcribe(self, video_path):
        try:
            result = self.whisper_model.transcribe(
                str(video_path),
                language=None,
                task="transcribe",
                verbose=False
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
                "failed": False
            }
        except Exception as e:
            print(f"  Whisper failed on {video_path}: {e}")
            return {
                "text": "",
                "confidence": -2.0,
                "language": "unknown",
                "reliable": False,
                "failed": True
            }

    def encode(self, transcript_result):
        text = transcript_result["text"]
        confidence = transcript_result["confidence"]
        if not text or confidence < -1.5:
            return torch.zeros(384, dtype=torch.float32)
        with torch.no_grad():
            embedding = self.sbert_model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
        embedding = embedding.float().detach().cpu()
        return embedding

    def project(self, embedding_384):
        with torch.no_grad():
            output = self.projection(embedding_384.detach())
        return output

    def forward(self, video_path):
        transcript_result = self.transcribe(video_path)
        embedding_384 = self.encode(transcript_result)
        embedding_256 = self.project(embedding_384)
        return embedding_256

    def extract_features(self, video_path):
        transcript_result = self.transcribe(video_path)
        embedding_384 = self.encode(transcript_result)
        return {
            "embedding_384": embedding_384,
            "transcript": transcript_result["text"],
            "confidence": transcript_result["confidence"],
            "reliable": transcript_result["reliable"]
        }

    def save_projection(self, path="saved_models/semantic_projection.pth"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.projection.state_dict(), path)
        print(f"Projection weights saved to {path}")

    def load_projection(self, path="saved_models/semantic_projection.pth"):
        self.projection.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Projection weights loaded from {path}")

    def get_projection(self):
        return self.projection