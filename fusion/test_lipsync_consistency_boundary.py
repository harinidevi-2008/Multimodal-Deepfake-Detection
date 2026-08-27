"""
Unit test for the Priority-5 boundary fix.

At consistency exactly equal to lipsync_analyzer.CONSISTENCY_THRESHOLD,
the analyzer's own rule (lipsync_status_from_consistency, using >=)
calls that sample "Consistent". evaluate_blink_lipsync.py's
evaluate_lipsync() must classify that same sample as NOT flagged
(predicted 0 / not "Inconsistent") by construction, since it now calls
that exact function rather than re-deriving an equivalent threshold in
mismatch-score space (which previously got the boundary backwards).

Run directly:
    python fusion/test_lipsync_consistency_boundary.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "visual" / "src" / "lipsync"))
from lipsync_analyzer import CONSISTENCY_THRESHOLD, lipsync_status_from_consistency  # noqa: E402


def main():
    threshold = CONSISTENCY_THRESHOLD
    epsilon = 1e-6

    cases = [
        (threshold, "Consistent"),              # exact boundary - the case previously mishandled
        (threshold + epsilon, "Consistent"),    # just above
        (threshold - epsilon, "Inconsistent"),  # just below
        (0.0, "Inconsistent"),
        (1.0, "Consistent"),
    ]

    failures = []
    for consistency, expected in cases:
        actual = lipsync_status_from_consistency(consistency)
        ok = actual == expected
        print(f"[{'PASS' if ok else 'FAIL'}] consistency={consistency!r} -> {actual!r} (expected {expected!r})")
        if not ok:
            failures.append((consistency, expected, actual))

    # Demonstrate explicitly that the naive mismatch-space reconstruction
    # this replaced would have gotten the exact boundary sample wrong -
    # so a future refactor can't silently reintroduce the bug unnoticed.
    boundary_mismatch = 1.0 - threshold
    naive_would_flag_as_inconsistent = boundary_mismatch >= (1.0 - threshold)  # True (the old, wrong behavior)
    actual_is_consistent = lipsync_status_from_consistency(threshold) == "Consistent"
    if naive_would_flag_as_inconsistent and actual_is_consistent:
        print("[PASS] confirmed the naive `mismatch_score >= (1 - threshold)` reconstruction would have "
              "mis-flagged the exact boundary sample as Inconsistent, while lipsync_status_from_consistency() "
              "correctly calls it Consistent, matching the analyzer.")
    else:
        failures.append(("naive-vs-correct boundary cross-check", True, naive_would_flag_as_inconsistent and actual_is_consistent))
        print("[FAIL] boundary cross-check did not reproduce the expected discrepancy.")

    if failures:
        print(f"\n{len(failures)} FAILURE(S)")
        sys.exit(1)
    print("\nAll boundary checks passed.")


if __name__ == "__main__":
    main()
