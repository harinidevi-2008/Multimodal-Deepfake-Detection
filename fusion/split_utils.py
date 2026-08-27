"""
Shared utilities for building and consuming the persistent train/validation/test
split (fusion/data_split.json).

Why this exists: train_fusion.py and train_enhanced_fusion.py previously each
called sklearn's train_test_split() fresh, on their own sample list, with a
fixed random_state. That is reproducible *within one script's own run*, but
nothing guaranteed two different scripts (a training script and an evaluation
script, or the enhanced trainer and a single-modality classifier trainer)
agreed on which relative paths landed in which split - the input ordering,
filtering, and even sklearn version can shift the result. This module makes
the split an artifact (fusion/data_split.json), computed once by
create_split.py, then loaded read-only by everything else.

Group-aware splitting: FakeAVCeleb organizes clips under
<category>/<ethnicity>/<gender>/id<NNNNN>/<filename>, and fake clips further
embed a SECOND identity token in the filename itself (the face/voice source
that was swapped in), e.g.

    FakeVideo-FakeAudio/African/men/id00076/00109_10_id00476_wavtolip.npy

lives under folder identity id00076 but was generated using id00476. Splitting
by folder identity alone would still let id00476-derived content appear in
one split while an id00476 real clip (if any) lands in another. To guard
against that, every id-like token found ANYWHERE in a sample's relative path
(folder names + filename) is treated as a group member, and a union-find
merges any two samples that share an id token into the same group. Whole
groups - never individual samples - are ever assigned to a split.

This is a heuristic, not a guarantee: it only catches leakage that is
detectable from an "id<digits>" token in the path/filename. If FakeAVCeleb
encodes identity some other way in a given subset, that leakage would not be
caught here - see create_split.py's printed report for how many samples fell
back to a singleton (path-only) group with no detected id token.
"""

import json
import re
import statistics
from collections import Counter
from pathlib import Path

ID_PATTERN = re.compile(r"id0*([0-9]+)", re.IGNORECASE)

DEFAULT_SPLIT_PATH = Path(__file__).resolve().parent / "data_split.json"

SPLIT_NAMES = ("train", "validation", "test")


def extract_group_ids(relative_path_str):
    """Return the set of identity tokens (as ints) referenced anywhere in a
    relative path string (folder names + filename)."""
    return set(int(m) for m in ID_PATTERN.findall(relative_path_str))


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def build_groups(relative_path_strs):
    """
    relative_path_strs: iterable of POSIX-style relative path strings.

    Returns: dict {relative_path_str: group_key}. Samples sharing any
    id-token end up under the same group_key (formatted "id:<root>").
    A sample with no id-like token anywhere in its path becomes its own
    singleton group, keyed by the path itself ("path:<relative_path>").
    """
    uf = UnionFind()
    path_ids = {}
    for rp in relative_path_strs:
        ids = extract_group_ids(rp)
        path_ids[rp] = ids
        ids = list(ids)
        for i in range(1, len(ids)):
            uf.union(ids[0], ids[i])

    groups = {}
    for rp, ids in path_ids.items():
        if ids:
            root = uf.find(next(iter(ids)))
            groups[rp] = f"id:{root}"
        else:
            groups[rp] = f"path:{rp}"
    return groups


def group_size_diagnostics(group_of, largest_group_share_warn_threshold=0.20):
    """
    Pure diagnostic over a {relative_path: group_key} mapping (see
    build_groups()) - reports how lumpy the union-find identity
    grouping turned out to be, WITHOUT changing the grouping itself.

    Why this matters: a FakeAVCeleb fake clip's filename embeds a
    SECOND identity token (the swapped-in source) alongside its own
    folder identity, so union-find can in principle chain many folder
    identities together through shared "hub" id tokens - in the worst
    case merging a large fraction of the whole dataset into one giant
    group that then has to be assigned to just a single split. This
    function surfaces that risk numerically (largest/median group
    size, singleton count, achieved share of the dataset) so it can be
    checked against real data, instead of either assuming the grouping
    is fine or blindly replacing it without evidence either way.

    Returns a dict with total_samples, total_groups, largest_group_size,
    largest_group_key, median_group_size, singleton_group_count,
    largest_group_share (largest_group_size / total_samples), and a
    'warning' string (or None) when largest_group_share exceeds
    largest_group_share_warn_threshold. This is a heuristic flag, not a
    hard failure: a legitimately large group (e.g. one very prolific
    source identity) is still handled correctly by construction - the
    warning only says it is worth a human look, never that grouping
    must be changed.
    """
    total_samples = len(group_of)
    sizes = Counter(group_of.values())
    total_groups = len(sizes)
    if total_groups == 0:
        return {
            "total_samples": 0, "total_groups": 0, "largest_group_size": 0,
            "largest_group_key": None, "median_group_size": 0,
            "singleton_group_count": 0, "largest_group_share": 0.0, "warning": None,
        }

    largest_group_key, largest_group_size = sizes.most_common(1)[0]
    median_group_size = statistics.median(sizes.values())
    singleton_group_count = sum(1 for s in sizes.values() if s == 1)
    largest_group_share = (largest_group_size / total_samples) if total_samples else 0.0

    warning = None
    if largest_group_share > largest_group_share_warn_threshold:
        warning = (
            f"Largest identity group ('{largest_group_key}') contains {largest_group_size} of "
            f"{total_samples} samples ({largest_group_share * 100:.1f}%) - above the "
            f"{largest_group_share_warn_threshold * 100:.0f}% heuristic threshold. This can happen "
            "legitimately (a genuinely prolific source identity chained many clips together via shared "
            "id tokens), but is worth a manual check that union-find isn't over-merging unrelated "
            "samples through a coincidental id-token collision."
        )

    return {
        "total_samples": total_samples,
        "total_groups": total_groups,
        "largest_group_size": largest_group_size,
        "largest_group_key": largest_group_key,
        "median_group_size": median_group_size,
        "singleton_group_count": singleton_group_count,
        "largest_group_share": round(largest_group_share, 4),
        "warning": warning,
    }


def to_key(relative_path):
    """Normalize a relative path (str or Path, either OS's separators) to the
    POSIX-style string used as the canonical key in data_split.json. This
    matters because this project runs on both Windows (backslashes) and
    Linux/Mac (forward slashes) machines across the team - without this,
    a split file written on one OS can silently fail to match on another."""
    return Path(relative_path).as_posix()


def load_split(split_path=DEFAULT_SPLIT_PATH):
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"No split file at {split_path}. Run `python fusion/create_split.py` "
            "first to generate the persistent train/validation/test split."
        )
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_of(relative_path, split_data):
    """Returns 'train' / 'validation' / 'test', or None if the path is not
    part of the saved split (e.g. it was added to the feature dirs after the
    split was generated - re-run create_split.py to pick it up)."""
    entry = split_data.get("samples", {}).get(to_key(relative_path))
    return entry["split"] if entry else None


def filter_relative_paths(relative_paths, split_data, split_name):
    """split_name: one of 'train' / 'validation' / 'test' / 'all'.
    'all' returns every input path unchanged (used only for explicitly
    labeled debugging runs, never as the default)."""
    if split_name == "all":
        return list(relative_paths)
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown split '{split_name}', expected one of {SPLIT_NAMES + ('all',)}")
    samples = split_data.get("samples", {})
    out = []
    for rp in relative_paths:
        entry = samples.get(to_key(rp))
        if entry is not None and entry["split"] == split_name:
            out.append(rp)
    return out


def verify_feature_files_exist(split_data, max_report=20):
    """
    Re-checks, against the actual filesystem, that every sample in the
    split still exists under all five feature roots recorded in the
    split's own metadata (visual_root/audio_root/semantic_root/
    blink_root/lipsync_root - saved by create_split.py). This is a
    defensive double-check on top of collect_aligned_samples()'s
    at-creation-time guarantee: it catches drift after the split was
    created (a file deleted/moved, a re-extraction that changed
    filenames, a hand-edited data_split.json) rather than only trusting
    that the JSON was correct when it was written.

    Returns {"checked": bool, "missing": [{"relative_path", "missing_from": [...]}]}
    "checked" is False (with an explanatory note, not an error) if the
    split's metadata doesn't record all five root paths - e.g. a
    hand-crafted or older split file - since there is nothing to check
    file existence against in that case.
    """
    metadata = split_data.get("metadata", {})
    root_keys = {
        "visual": "visual_root", "audio": "audio_root", "semantic": "semantic_root",
        "blink": "blink_root", "lipsync": "lipsync_root",
    }
    roots = {}
    for name, key in root_keys.items():
        value = metadata.get(key)
        if not value:
            return {"checked": False, "missing": [],
                    "note": f"split metadata has no '{key}' - cannot re-check file existence."}
        roots[name] = Path(value)

    missing = []
    for rp, entry in split_data.get("samples", {}).items():
        missing_from = [name for name, root in roots.items() if not (root / rp).exists()]
        if missing_from:
            missing.append({"relative_path": rp, "missing_from": missing_from})
            if len(missing) >= max_report:
                break

    return {"checked": True, "missing": missing, "roots": {k: str(v) for k, v in roots.items()}}


def verify_split(split_data, group_key_fn=None, check_file_existence=True):
    """
    Returns a report dict. Checks:

    1. Flat disjointness: no relative path key is assigned to more than
       one split. Trivially guaranteed by the JSON's own structure (one
       key -> one split value) but checked explicitly anyway.
    2. Group integrity (the check that actually matters for leakage):
       every sample sharing a group with another sample must be in the
       SAME split. This is the real guarantee create_split.py provides.
    3. Class balance: every split must contain BOTH real and fake
       samples - a split with only one class can't train a classifier
       or compute a meaningful precision/recall/F1/ROC-AUC on it, so
       this is treated as a hard failure, not a warning.
    4. File existence: every sample still exists under all five
       recorded feature roots (see verify_feature_files_exist above) -
       skipped (not failed) if the split's metadata doesn't record all
       five roots.
    """
    samples = split_data.get("samples", {})
    by_split = {name: [] for name in SPLIT_NAMES}
    for rp, entry in samples.items():
        by_split.setdefault(entry["split"], []).append(rp)

    sets = {name: set(paths) for name, paths in by_split.items()}
    flat_overlaps = {}
    names = list(sets.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = sets[names[i]] & sets[names[j]]
            if overlap:
                flat_overlaps[f"{names[i]}∩{names[j]}"] = len(overlap)

    group_violations = []
    if group_key_fn is None:
        group_key_fn = lambda rp: split_data["samples"][rp].get("group")
    group_to_splits = {}
    for rp in samples:
        g = group_key_fn(rp)
        group_to_splits.setdefault(g, set()).add(samples[rp]["split"])
    for g, split_set in group_to_splits.items():
        if len(split_set) > 1:
            group_violations.append({"group": g, "splits": sorted(split_set)})

    label_counts = {name: {"real": 0, "fake": 0} for name in SPLIT_NAMES}
    for rp, entry in samples.items():
        bucket = label_counts.setdefault(entry["split"], {"real": 0, "fake": 0})
        bucket["real" if entry["label"] == 0 else "fake"] += 1

    label_distribution = {}
    class_violations = []
    for name in SPLIT_NAMES:
        counts = label_counts.get(name, {"real": 0, "fake": 0})
        total = counts["real"] + counts["fake"]
        fake_pct = (counts["fake"] / total * 100) if total else 0.0
        label_distribution[name] = {**counts, "total": total, "fake_percentage": round(fake_pct, 2)}
        if total > 0 and (counts["real"] == 0 or counts["fake"] == 0):
            class_violations.append({"split": name, **counts})

    file_existence = (
        verify_feature_files_exist(split_data) if check_file_existence
        else {"checked": False, "missing": [], "note": "check_file_existence=False"}
    )
    missing_files = file_existence.get("missing", [])

    ok = (
        not flat_overlaps
        and not group_violations
        and not class_violations
        and not missing_files
    )

    return {
        "total_samples": len(samples),
        "counts_per_split": {name: len(paths) for name, paths in by_split.items()},
        "label_distribution": label_distribution,
        "flat_overlaps": flat_overlaps,
        "group_violations": group_violations,
        "class_violations": class_violations,
        "file_existence": file_existence,
        "ok": ok,
    }
