from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ROOT = Path(r"C:\LAV-DF\dev")

STREAMS = {
    "visual": (ROOT / "visual" / "data" / "features_aligned_lavdf", (1280,)),
    "audio": (ROOT / "audio" / "data" / "features_lavdf", (768,)),
    "blink": (ROOT / "visual" / "data" / "blink_features_lavdf", (4,)),
    "lipsync": (ROOT / "visual" / "data" / "lipsync_features_lavdf", (2,)),
    "semantic": (ROOT / "semantic" / "data" / "features_lavdf", (384,)),
}

def feature_files(folder: Path) -> dict[str, Path]:
    # Ignore temporary loader-test files such as _loader_test_004053.npy.
    return {
        p.stem: p
        for p in folder.glob("*.npy")
        if not p.stem.startswith("_")
    }

def main() -> int:
    report: dict = {
        "project_root": str(ROOT),
        "video_root": str(VIDEO_ROOT),
        "streams": {},
        "alignment": {},
        "overall_pass": False,
    }

    sets: dict[str, set[str]] = {}
    stream_passes: list[bool] = []

    for name, (folder, expected_shape) in STREAMS.items():
        files = feature_files(folder)
        sets[name] = set(files)
        malformed: list[dict] = []

        for sample_id, path in files.items():
            try:
                arr = np.load(path, allow_pickle=False)
                reasons = []
                if arr.shape != expected_shape:
                    reasons.append(f"shape {arr.shape}, expected {expected_shape}")
                if arr.dtype != np.float32:
                    reasons.append(f"dtype {arr.dtype}, expected float32")
                if not np.isfinite(arr).all():
                    reasons.append("contains NaN or Inf")
                if reasons:
                    malformed.append({"id": sample_id, "file": str(path), "issues": reasons})
            except Exception as exc:
                malformed.append({"id": sample_id, "file": str(path), "issues": [repr(exc)]})

        stream_ok = len(files) >= 100 and not malformed
        stream_passes.append(stream_ok)
        report["streams"][name] = {
            "folder": str(folder),
            "count": len(files),
            "expected_shape": list(expected_shape),
            "malformed_count": len(malformed),
            "malformed_examples": malformed[:10],
            "pass": stream_ok,
        }

    all_ids = set.union(*sets.values()) if sets else set()
    common_ids = set.intersection(*sets.values()) if sets else set()
    report["alignment"] = {
        "unique_feature_ids_across_streams": len(all_ids),
        "ids_present_in_all_five_streams": len(common_ids),
        "missing_by_stream": {
            name: sorted(all_ids - ids)[:30] for name, ids in sets.items()
        },
        "common_ids_without_source_video": sorted(
            sample_id for sample_id in common_ids
            if not (VIDEO_ROOT / f"{sample_id}.mp4").exists()
        )[:30],
        "pass": len(common_ids) >= 100 and not any(
            not (VIDEO_ROOT / f"{sample_id}.mp4").exists() for sample_id in common_ids
        ),
    }

    report["overall_pass"] = all(stream_passes) and report["alignment"]["pass"]

    output_path = ROOT / "scripts" / "lavdf_100_validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== LAV-DF FEATURE VALIDATION ==========")
    for name, info in report["streams"].items():
        status = "PASS" if info["pass"] else "CHECK"
        print(
            f"{name:8}: {status} | files={info['count']} | "
            f"malformed={info['malformed_count']} | expected_shape={tuple(info['expected_shape'])}"
        )
        for item in info["malformed_examples"][:3]:
            print(f"          {item['id']}: {', '.join(item['issues'])}")

    print(
        f"Alignment: {'PASS' if report['alignment']['pass'] else 'CHECK'} | "
        f"shared IDs across all five streams={report['alignment']['ids_present_in_all_five_streams']}"
    )
    if report["alignment"]["missing_by_stream"]:
        for name, missing in report["alignment"]["missing_by_stream"].items():
            if missing:
                print(f"  IDs missing from {name} (first 10): {missing[:10]}")
    if report["alignment"]["common_ids_without_source_video"]:
        print("  Shared feature IDs without source MP4 (first 10):",
              report["alignment"]["common_ids_without_source_video"][:10])

    print(f"Report: {output_path}")
    print("OVERALL:", "PASS" if report["overall_pass"] else "CHECK REQUIRED")
    return 0 if report["overall_pass"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
