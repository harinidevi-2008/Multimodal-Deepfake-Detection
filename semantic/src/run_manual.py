"""
Manual real-model integration script for the semantic stream.

This is NOT a synthetic pytest unit test: it loads the real Whisper +
Sentence-BERT models via SemanticStream and downloads model weights on
first run. It is meant to be run directly, not collected by pytest.

Renamed from run_test.py to run_manual.py so pytest's default
`test_*.py` / `*_test.py` collection patterns no longer match this
file at all. This script was previously top-level module code, which
meant plain `pytest` collection imported and RAN it (including a
network model download) just by discovering the file - a prior pass
wrapped it in `main()` behind the `if __name__ == "__main__":` guard,
but pytest still *imports* any file matching its test-file naming
pattern to look for test functions inside it, so the file itself still
needed to stop matching that pattern. Nothing about what this script
does when you run it directly has changed.

Run directly to exercise it:
    python semantic/src/run_manual.py
"""

import torch
from semantic_stream import SemanticStream
from sentence_transformers import util


def main():
    print("Loading models...")
    stream = SemanticStream(whisper_size="base")

    print("Running test...\n")

    sentence = "The prime minister announced his resignation today."

    # Step 1 - encode a sentence
    with torch.no_grad():
        embedding = stream.sbert_model.encode(
            sentence,
            convert_to_tensor=True,
            normalize_embeddings=True
        ).float().detach().cpu()

    print(f"Sentence    : {sentence}")
    print(f"Shape       : {embedding.shape}")
    print(f"Dtype       : {embedding.dtype}")
    print(f"Norm        : {embedding.norm().item():.4f}")
    print()

    # Step 2 - project to 256
    output = stream.project(embedding)
    print(f"After projection shape : {output.shape}")
    print(f"After projection dtype : {output.dtype}")
    print()

    # Step 3 - check no errors
    assert output.shape == torch.Size([256]), "WRONG SHAPE"
    assert output.dtype == torch.float32, "WRONG DTYPE"
    assert not torch.isnan(output).any(), "NaN FOUND"
    assert not torch.isinf(output).any(), "INF FOUND"

    # Step 4 - test similarity between two sentences
    print("Testing semantic similarity...")
    with torch.no_grad():
        s1 = stream.sbert_model.encode(
            "He resigned from his position",
            convert_to_tensor=True,
            normalize_embeddings=True
        ).float().detach().cpu()

        s2 = stream.sbert_model.encode(
            "The man announced he is quitting his job",
            convert_to_tensor=True,
            normalize_embeddings=True
        ).float().detach().cpu()

        s3 = stream.sbert_model.encode(
            "I enjoy eating pizza on weekends",
            convert_to_tensor=True,
            normalize_embeddings=True
        ).float().detach().cpu()

    sim_same = util.cos_sim(s1, s2).item()
    sim_diff = util.cos_sim(s1, s3).item()

    print(f"Same topic similarity : {sim_same:.3f}  (must be above 0.6)")
    print(f"Diff topic similarity : {sim_diff:.3f}  (must be below 0.4)")
    print()

    assert sim_same > 0.6, f"Same topic similarity too low: {sim_same:.3f}"
    assert sim_diff < 0.4, f"Different topic similarity too high: {sim_diff:.3f}"

    print("=" * 45)
    print("ALL TESTS PASSED")
    print("Semantic stream is fully working.")
    print(f"Output shape : {output.shape}")
    print(f"Output dtype : {output.dtype}")
    print("This is what your fusion teammate receives.")
    print("=" * 45)


if __name__ == "__main__":
    main()
