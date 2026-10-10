"""Small executable serializer smoke test for the public three-stream contract."""

from test_fake_probability_regression import test_serializer_keeps_fused_fake_probability_not_real_probability


if __name__ == "__main__":
    test_serializer_keeps_fused_fake_probability_not_real_probability()
    print("Serializer returned the fused probability and excluded five-stream diagnostics.")
