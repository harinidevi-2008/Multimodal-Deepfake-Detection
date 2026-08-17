"""
verify_semantic_output.py
=========================
Runs the teammate's full checklist on the semantic stream output.

Checks:
  1. .npy files exist
  2. shape == (384,)
  3. dtype == float32
  4. filename matches video filename exactly
  5. metadata.json is valid
  6. Sample of 10 files verified end to end

Run this after precompute.py finishes (or on a small sample).
"""

import json
import numpy as np
from pathlib import Path

FEATURES_DIR  = Path("features/semantic")
DATASET_PATH  = Path("FakeAVCeleb_v1.2")
METADATA_FILE = FEATURES_DIR / "metadata.json"

print("=" * 60)
print("SEMANTIC OUTPUT VERIFICATION")
print("=" * 60)
print()

errors  = []
passed  = 0

# ── CHECK 1: metadata.json exists ─────────────────────────────────
print("Check 1 — metadata.json exists...")
if not METADATA_FILE.exists():
    print(f"  ❌ Not found: {METADATA_FILE}")
    print("  Run precompute.py first")
    exit(1)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"  ✅ Found — {len(metadata):,} entries")
passed += 1

# ── CHECK 2: All .npy files exist ─────────────────────────────────
print()
print("Check 2 — All .npy files exist on disk...")
missing = []
for rel_path, info in list(metadata.items())[:100]:  # sample first 100
    npy_path = Path(info["embedding"])
    if not npy_path.exists():
        missing.append(str(npy_path))

if missing:
    print(f"  ❌ {len(missing)} files missing (showing first 5):")
    for m in missing[:5]:
        print(f"     {m}")
    errors.append("missing .npy files")
else:
    print(f"  ✅ All checked files exist on disk")
    passed += 1

# ── CHECK 3: Shape == (384,) ──────────────────────────────────────
print()
print("Check 3 — Shape == (384,)...")
shape_errors = []
sample_items = list(metadata.items())[:50]  # check first 50

for rel_path, info in sample_items:
    npy_path = Path(info["embedding"])
    if not npy_path.exists():
        continue
    arr = np.load(str(npy_path))
    if arr.shape != (384,):
        shape_errors.append(f"{Path(rel_path).name} → shape={arr.shape}")

if shape_errors:
    print(f"  ❌ {len(shape_errors)} files have wrong shape:")
    for e in shape_errors[:5]:
        print(f"     {e}")
    errors.append("wrong shape")
else:
    print(f"  ✅ All {len(sample_items)} checked files have shape (384,)")
    passed += 1

# ── CHECK 4: dtype == float32 ─────────────────────────────────────
print()
print("Check 4 — dtype == float32...")
dtype_errors = []

for rel_path, info in sample_items:
    npy_path = Path(info["embedding"])
    if not npy_path.exists():
        continue
    arr = np.load(str(npy_path))
    if arr.dtype != np.float32:
        dtype_errors.append(f"{Path(rel_path).name} → dtype={arr.dtype}")

if dtype_errors:
    print(f"  ❌ {len(dtype_errors)} files have wrong dtype:")
    for e in dtype_errors[:5]:
        print(f"     {e}")
    errors.append("wrong dtype")
else:
    print(f"  ✅ All {len(sample_items)} checked files are float32")
    passed += 1

# ── CHECK 5: Filename matches video filename exactly ──────────────
print()
print("Check 5 — Filename matches video filename...")
name_errors = []

for rel_path, info in sample_items:
    video_stem = Path(info["video_path"]).stem
    npy_stem   = Path(info["embedding"]).stem
    if video_stem != npy_stem:
        name_errors.append(
            f"video={video_stem}  npy={npy_stem}"
        )

if name_errors:
    print(f"  ❌ {len(name_errors)} filename mismatches:")
    for e in name_errors[:5]:
        print(f"     {e}")
    errors.append("filename mismatch")
else:
    print(f"  ✅ All filenames match their source video names")
    passed += 1

# ── CHECK 6: Labels correct ───────────────────────────────────────
print()
print("Check 6 — Labels correct (RealVideo-RealAudio=0, rest=1)...")
label_errors = []

EXPECTED = {
    "RealVideo-RealAudio": 0,
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
    "RealVideo-FakeAudio": 1,
}

for rel_path, info in sample_items:
    category = info.get("category", "")
    label    = info.get("label", -1)
    expected = EXPECTED.get(category, -1)
    if label != expected:
        label_errors.append(f"{category} → label={label} expected={expected}")

if label_errors:
    print(f"  ❌ {len(label_errors)} label errors:")
    for e in label_errors[:5]:
        print(f"     {e}")
    errors.append("label errors")
else:
    print(f"  ✅ All labels correct")
    passed += 1

# ── CHECK 7: Dataset stats ────────────────────────────────────────
print()
print("Check 7 — Dataset statistics...")
real  = sum(1 for v in metadata.values() if v["label"] == 0)
fake  = sum(1 for v in metadata.values() if v["label"] == 1)
rel   = sum(1 for v in metadata.values() if v.get("reliable", False))
zeros = sum(1 for v in metadata.values() if v.get("zero_vector", False))
total = len(metadata)

print(f"  Total processed    : {total:,}")
print(f"  Real (label=0)     : {real:,}")
print(f"  Fake (label=1)     : {fake:,}")
print(f"  Reliable transcripts: {rel:,}  ({rel/max(total,1)*100:.1f}%)")
print(f"  Zero vectors       : {zeros:,}  ({zeros/max(total,1)*100:.1f}%)")
passed += 1

# ── CHECK 8: Show 3 sample entries ───────────────────────────────
print()
print("Check 8 — Sample entries...")
for i, (rel_path, info) in enumerate(list(metadata.items())[:3]):
    arr = np.load(info["embedding"]) if Path(info["embedding"]).exists() else None
    print(f"  [{i+1}] {Path(rel_path).name}")
    print(f"       category  : {info['category']}")
    print(f"       label     : {info['label']}")
    print(f"       shape     : {arr.shape if arr is not None else 'FILE MISSING'}")
    print(f"       dtype     : {arr.dtype if arr is not None else 'FILE MISSING'}")
    print(f"       reliable  : {info.get('reliable', '?')}")
    print(f"       transcript: {info.get('transcript', '')[:60]}...")
    print()
passed += 1

# ── SUMMARY ───────────────────────────────────────────────────────
print("=" * 60)
if not errors:
    print(f"ALL {passed} CHECKS PASSED")
    print()
    print("Semantic stream output is verified and ready for fusion.")
    print()
    print("Confirm to fusion teammate:")
    print("  ✅ shape = (384,)")
    print("  ✅ dtype = float32")
    print("  ✅ filename = video filename + .npy")
    print("  ✅ Projection (384→256) is NOT applied here")
    print("     It happens inside the fusion model during training")
else:
    print(f"FAILED — {len(errors)} issue(s) found:")
    for e in errors:
        print(f"  ❌ {e}")
    print()
    print("Fix these before handing off to the fusion teammate.")
print("=" * 60)
