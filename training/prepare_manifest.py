"""Generate reproducible partitions without splitting scene and duplicate groups."""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from backend.app.preprocess import CLASSES
from training.preprocess import candidates, group_keys


def make_splits(rows, seed, cambodia_min_groups):
    """Assign every row to train / validation / test (and optionally cambodia_holdout).

    Steps:
      1. Join rows that must stay together (same scene, same file, near-duplicate) into groups.
      2. Keep fixed groups where they belong (source test images -> test, train-only -> train).
      3. Optionally set aside ~20% of Cambodian groups as a separate local holdout.
      4. Place the remaining groups one by one into the split that keeps the class
         balance closest to the target fractions (70/15/15, or 85/15 if a test set already exists).
    """
    # ---- Step 1: union-find ("disjoint set") to merge connected rows into groups ----
    # parent[i] points towards the representative ("root") row of i's group.
    parent = list(range(len(rows)))

    def root(index):
        # Follow parent links up to the root; also shortens the path for later calls.
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}  # group key -> first row index that had this key
    for index, row in enumerate(rows):
        for key in group_keys(row):
            # Merge this row's group with the group of the first row that shared the key.
            parent[root(index)] = root(seen.setdefault(key, index))

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)

    # ---- Step 2: sort groups into categories ----
    local, pending, fixed, train_only = [], [], [], []
    for group in groups.values():
        source_splits = {row.get("original_split") for row in group}
        if any(row.get("split_policy") == "train_only" for row in group):
            if source_splits & {"test", "non_train"}:
                raise ValueError("A train-only group crosses an existing source test boundary.")
            train_only.extend(group)
        elif source_splits & {"test", "non_train"}:
            if source_splits & {"train", "val", "validation"}:
                raise ValueError("A scene or duplicate group crosses an existing source test boundary. Resolve the overlap first.")
            fixed.extend(group)
        elif all(row.get("country", "").strip().lower() in {"cambodia", "kh"} for row in group):
            local.append(group)
        else:
            pending.append(group)

    # ---- Step 3: optional Cambodian holdout ----
    rng = random.Random(seed)  # seeded, so the same seed always gives the same split
    rng.shuffle(local)
    holdout = []
    if len(local) >= max(cambodia_min_groups, 3):
        # Require all classes in local evaluation and leave each represented locally.
        target = max(1, round(len(local) * 0.2))
        def labels(groups):
            return {r["verified_quality_label"] for group in groups for r in group}

        for _ in range(64):
            if labels(local[:target]) == labels(local[target:]) == set(CLASSES):
                holdout, local = local[:target], local[target:]
                break
            rng.shuffle(local)
    # ---- Step 4: greedy, class-balanced assignment ----
    pending.extend(local)
    rng.shuffle(pending)
    # Place the biggest groups first; small groups fill the gaps at the end.
    pending.sort(key=len, reverse=True)

    totals = Counter(r["verified_quality_label"] for group in pending for r in group)
    partitions = {"train": train_only, "validation": [], "test": fixed}
    fractions = {"train": 0.85, "validation": 0.15} if fixed else {
        "train": 0.7, "validation": 0.15, "test": 0.15}
    counts = {name: Counter() for name in fractions}
    for group in pending:
        labels = Counter(r["verified_quality_label"] for r in group)
        # How much worse the class balance of `partition` would get by adding this group.
        # For each label: (new distance from target)^2 - (old distance)^2, scaled by the target size.
        # The group goes to the split with the smallest penalty.
        def penalty(partition):
            count = counts[partition]
            fraction = fractions[partition]
            return sum(
                ((count[label] + n - totals[label] * fraction) ** 2
                 - (count[label] - totals[label] * fraction) ** 2) / max(totals[label] * fraction, 1)
                for label, n in labels.items()
            )
        split = min(fractions, key=penalty)
        partitions[split].extend(group)
        counts[split].update(labels)
    partitions["cambodia_holdout"] = [r for group in holdout for r in group]
    for split in ("train", "validation", "test"):
        if {r["verified_quality_label"] for r in partitions[split]} != set(CLASSES):
            raise ValueError(f"Grouped split {split} lacks class coverage. Review group sizes and label balance.")
    # Return a copy of every row with a new "split" column added.
    return [dict(row, split=split) for split, group in partitions.items() for row in group]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cambodia-min-groups", type=int, default=10)
    args = parser.parse_args()
    try:
        rows = make_splits(candidates(args.manifest), args.seed, args.cambodia_min_groups)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(Counter((r["split"], r["verified_quality_label"]) for r in rows))


if __name__ == "__main__":
    main()
