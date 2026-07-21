from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2Model

print("Loading Processor...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("Processor Loaded")

print("Loading Model...")

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base-960h"
)

print("Model Loaded Successfully")