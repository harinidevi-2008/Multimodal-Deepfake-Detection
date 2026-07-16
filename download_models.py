print("Starting model downloads...")
print()

print("Downloading Whisper base model - about 150 MB")
print("This will take a few minutes...")
import whisper
model = whisper.load_model("base")
print("Whisper downloaded successfully")
print()

print("Downloading Sentence-BERT model - about 90 MB")
print("This will take a few minutes...")
from sentence_transformers import SentenceTransformer
sbert = SentenceTransformer("all-MiniLM-L6-v2")
print("Sentence-BERT downloaded successfully")
print()

print("Both models are now saved on your machine")
print("They will never be downloaded again")