"""Generate reproducible partitions without splitting scene and duplicate groups."""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

from backend.app.preprocess import CLASSES
from training.preprocess import candidates, group_keys


def make_splits(rows, seed, cambodia_min_groups):
    parent = list(range(len(rows)))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    for index, row in enumerate(rows):
        for key in group_keys(row):
            parent[root(index)] = root(seen.setdefault(key, index))

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)

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

    rng = random.Random(seed)
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
    pending.extend(local)
    rng.shuffle(pending)
    pending.sort(key=len, reverse=True)

    totals = Counter(r["verified_quality_label"] for group in pending for r in group)
    partitions = {"train": train_only, "validation": [], "test": fixed}
    fractions = {"train": 0.85, "validation": 0.15} if fixed else {
        "train": 0.7, "validation": 0.15, "test": 0.15}
    counts = {name: Counter() for name in fractions}
    for group in pending:
        labels = Counter(r["verified_quality_label"] for r in group)
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
