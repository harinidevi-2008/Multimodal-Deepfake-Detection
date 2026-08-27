"""
Regression guard for the "no checkpoint saved when the first
validation epoch scores exactly F1=0.0" bug fixed in this hardening
pass.

The bug: `best_f1 = 0.0` followed by `if val_f1 > best_f1: save()`
means a first epoch with val_f1 == 0.0 never saves anything (0.0 > 0.0
is False) - a fully completed training run can finish with zero
checkpoints on disk. The fix is to start below any achievable F1 (this
repo uses -1.0) so the comparison is unconditionally True on the first
epoch, regardless of how bad that first epoch's F1 happens to be.

This test does two independent things, deliberately without running
real training (no dataset needed here):

1. Demonstrates the logic itself: with best_f1 = -1.0, a val_f1 of
   exactly 0.0 registers as "improved"; with the old best_f1 = 0.0, the
   same val_f1 does NOT register as improved - proving the fix actually
   changes the outcome for this exact edge case, not just cosmetically.
2. Greps the three training scripts' source to confirm none of them
   have regressed back to initializing their best-F1 tracker at 0.0 -
   a static guard so a future refactor can't silently reintroduce the
   bug without this test failing.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TRAINING_SCRIPTS = [
    REPO_ROOT / "classifiers" / "train_classifier.py",
    REPO_ROOT / "fusion" / "train_fusion.py",
    REPO_ROOT / "fusion" / "train_enhanced_fusion.py",
]

# Matches `best_f1 = 0.0` / `best_val_f1 = 0.0` (the buggy pattern),
# tolerant of whitespace variants.
BUGGY_INIT_PATTERN = re.compile(r"\bbest_(?:val_)?f1\s*=\s*0\.0\b")
# Matches the fixed pattern, used to confirm the fix is actually present
# (not just that the buggy pattern is absent - e.g. if the variable were
# renamed entirely, both checks should still make sense together).
FIXED_INIT_PATTERN = re.compile(r"\bbest_(?:val_)?f1\s*=\s*-1\.0\b")


def check_logic():
    """Part 1: the comparison itself, with both the fixed and the old
    starting values, on the exact boundary case (val_f1 == 0.0)."""
    val_f1 = 0.0

    best_f1_fixed = -1.0
    improved_with_fix = val_f1 > best_f1_fixed

    best_f1_old_buggy = 0.0
    improved_with_old_bug = val_f1 > best_f1_old_buggy

    ok = True
    if improved_with_fix is not True:
        print(f"[FAIL] With best_f1=-1.0, val_f1=0.0 should register as improved (got {improved_with_fix}).")
        ok = False
    else:
        print("[PASS] With best_f1=-1.0, a first-epoch val_f1=0.0 correctly registers as improved "
              "(a checkpoint would be saved).")

    if improved_with_old_bug is not False:
        print(f"[FAIL] Sanity check failed: expected the OLD best_f1=0.0 pattern to NOT register "
              f"val_f1=0.0 as improved (got {improved_with_old_bug}); the demonstration is invalid.")
        ok = False
    else:
        print("[PASS] Confirmed the OLD best_f1=0.0 pattern would have skipped saving a checkpoint "
              "for this exact case - this is the bug this pass fixed.")

    return ok


def check_source_guards():
    """Part 2: static regression guard against reintroducing best_f1/best_val_f1 = 0.0."""
    ok = True
    for script in TRAINING_SCRIPTS:
        if not script.exists():
            print(f"[FAIL] Expected training script not found: {script}")
            ok = False
            continue

        text = script.read_text(encoding="utf-8")
        buggy_matches = BUGGY_INIT_PATTERN.findall(text)
        fixed_matches = FIXED_INIT_PATTERN.findall(text)

        if buggy_matches:
            print(f"[FAIL] {script.name}: found the buggy 'best_f1 = 0.0' / 'best_val_f1 = 0.0' "
                  f"initialization pattern ({len(buggy_matches)} occurrence(s)) - this would silently "
                  "skip saving a checkpoint on a first epoch with val_f1 exactly 0.0.")
            ok = False
        elif not fixed_matches:
            print(f"[FAIL] {script.name}: neither the buggy nor the expected fixed "
                  "'best_f1 = -1.0' / 'best_val_f1 = -1.0' pattern was found - the initialization "
                  "logic may have changed shape; update this guard if so.")
            ok = False
        else:
            print(f"[PASS] {script.name}: initializes its best-F1 tracker below 0.0 "
                  f"({len(fixed_matches)} occurrence(s)), not at the buggy 0.0.")

    return ok


def main():
    logic_ok = check_logic()
    source_ok = check_source_guards()

    if logic_ok and source_ok:
        print("\nAll checkpoint-at-F1=0 checks passed.")
        sys.exit(0)
    else:
        print("\nOne or more checkpoint-at-F1=0 checks FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
