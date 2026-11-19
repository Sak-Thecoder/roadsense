"""Validation-only analysis of saved predictions from investigation.collect_predictions.

Prints, for every run/model: accuracy and per-class F1, a per-surface confusion
matrix, the effect of bias calibration (estimated fairly with cross-validation),
ensemble results, and logistic-regression heads trained on the saved features.
The test set is never used, so these choices don't "peek" at the final score.
"""
import argparse
import csv
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from backend.app.preprocess import CLASSES, MODELS

# StreetSurfaceVis 5-level rating -> our 3 grades.
GRADE = {"excellent": "Good", "good": "Good", "intermediate": "Moderate", "bad": "Poor", "very_bad": "Poor"}


def load(run_dir, name, split, rows_by_id):
    """Load saved (features, probabilities, manifest rows) for one model and split."""
    data = np.load(run_dir / f"{name}-{split}.npz")
    rows = [rows_by_id[i] for i in data["image_ids"]]
    return data["features"], data["probabilities"], rows


def labels(rows):
    """True grades as integers: 0 = Good, 1 = Moderate, 2 = Poor."""
    return np.array([CLASSES.index(r["verified_quality_label"]) for r in rows])


def summary(actual, predicted):
    """One-line text: accuracy, macro F1 and the F1 of each grade."""
    f1 = f1_score(actual, predicted, labels=range(3), average=None, zero_division=0)
    return (f"acc {accuracy_score(actual, predicted):.3f} macroF1 {f1.mean():.3f} "
            f"F1[G/M/P] {f1[0]:.2f}/{f1[1]:.2f}/{f1[2]:.2f}")


def surface_breakdown(actual, predicted, rows):
    """Print accuracy and confusion matrix separately for each surface type (asphalt, sett, ...)."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["surface_type"] or "cambodian"].append(index)
    for surface, indices in sorted(groups.items()):
        a, p = actual[indices], predicted[indices]
        print(f"    {surface:14} n={len(indices):4} acc {accuracy_score(a, p):.3f} "
              f"cm {confusion_matrix(a, p, labels=range(3)).tolist()}")


def fit_head(train_x, train_y, groups, val_x, weight, c=1.0):
    """Logistic-regression head on frozen pooled features; fine labels are mapped to the three grades."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Standardise features (mean 0, std 1) so logistic regression trains well.
    scaler = StandardScaler().fit(train_x)
    classes = sorted(set(train_y))
    counts = Counter(train_y)
    sample_weight = None
    if weight:
        # Balanced weights: each sample weighs total / (3 * size of its grade).
        grade_counts = Counter(groups[c] for c in train_y)
        sample_weight = np.array([len(train_y) / (3 * grade_counts[groups[y]]) for y in train_y])
    head = LogisticRegression(C=c, max_iter=3000).fit(scaler.transform(train_x), train_y, sample_weight=sample_weight)
    fine = head.predict_proba(scaler.transform(val_x))
    # Add up fine-class probabilities into the three grades.
    grades = np.zeros((len(val_x), 3))
    for column, cls in enumerate(head.classes_):
        grades[:, CLASSES.index(groups[cls])] += fine[:, column]
    del classes, counts
    return grades


def bias_search(probabilities, actual, grid=np.linspace(-1.5, 1.5, 31)):
    """Log-prior offsets for Moderate and Poor (Good fixed at 0) maximising macro F1.

    Adding a constant to log(probability) of a class makes the model pick that class
    more (positive) or less (negative) often. We try every pair on a 31x31 grid.
    Returns (best_macro_f1, (moderate_offset, poor_offset)).
    """
    # clip avoids log(0) = -infinity.
    logs = np.log(np.clip(probabilities, 1e-7, 1))
    best = (-1, (0, 0))
    for m, p in itertools.product(grid, grid):
        score = f1_score(actual, (logs + [0, m, p]).argmax(1), average="macro")
        if score > best[0]:
            best = (score, (m, p))
    return best


def cross_validated_bias(probabilities, actual, rows, folds=5, seed=0):
    """Estimate the benefit of bias calibration without scoring on the rows used to fit it (grouped folds).

    The validation set is split into 5 folds by capture_group (whole scenes stay together).
    For each fold, offsets are tuned on the other 4 folds and applied to this one, so every
    prediction is made with offsets that never saw that image. This is an honest estimate.
    """
    rng = np.random.default_rng(seed)
    groups = sorted({r["capture_group"] for r in rows})
    fold_of = {g: int(f) for g, f in zip(groups, rng.integers(0, folds, len(groups)))}
    fold = np.array([fold_of[r["capture_group"]] for r in rows])
    predicted = np.zeros_like(actual)
    for k in range(folds):
        fit, held = fold != k, fold == k
        _, (m, p) = bias_search(probabilities[fit], actual[fit])
        predicted[held] = (np.log(np.clip(probabilities[held], 1e-7, 1)) + [0, m, p]).argmax(1)
    return predicted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as file:
        rows_by_id = {r["image_id"]: r for r in csv.DictReader(file)}

    # ---- Single models: overall, per surface, and with cross-validated bias calibration ----
    stored = {}
    for run, name in itertools.product(args.runs, MODELS):
        _, probs, rows = load(run, name, "validation", rows_by_id)
        stored[(run.name, name)] = probs
        actual = labels(rows)
        print(f"{run.name}/{name}: {summary(actual, probs.argmax(1))}")
        surface_breakdown(actual, probs.argmax(1), rows)
        cv = cross_validated_bias(probs, actual, rows)
        print(f"    bias-calibrated (5-fold grouped CV on validation): {summary(actual, cv)}")

    # ---- Ensembles: average the probabilities of 2, 3 or all models ----
    print("\nEnsembles (mean probability):")
    keys = list(stored)
    for size in (2, 3, len(keys)):
        for combo in itertools.combinations(keys, size):
            # Print the all-models ensemble only once.
            if size == len(keys) and combo != tuple(keys):
                continue
            probs = np.mean([stored[k] for k in combo], axis=0)
            print(f"  {' + '.join(f'{r}/{n}' for r, n in combo)}: {summary(actual, probs.argmax(1))}")

    # ---- Does training on finer labels help? Fit simple heads on the saved feature vectors ----
    print("\nFine-label heads on frozen fine-tuned features (train -> validation):")
    for run, name in itertools.product(args.runs, MODELS):
        train_x, _, train_rows = load(run, name, "train", rows_by_id)
        val_x, _, val_rows = load(run, name, "validation", rows_by_id)
        actual = labels(val_rows)
        for scheme in ("grade", "five_level", "surface_x_level"):
            # Label for this scheme: "Good", "excellent", or "asphalt:excellent".
            def fine(row):
                grade = row["verified_quality_label"]
                original = row["original_label"] if row["source"] == "streetSurfaceVis" else grade.lower()
                if scheme == "grade":
                    return grade
                if scheme == "five_level":
                    return original
                return f"{row['surface_type'] or 'cambodian'}:{original}"
            train_y = [fine(r) for r in train_rows]
            mapping = {f: r["verified_quality_label"] for f, r in zip(train_y, train_rows)}
            grades = fit_head(train_x, train_y, mapping, val_x, weight=True)
            print(f"  {run.name}/{name} {scheme:16} {summary(actual, grades.argmax(1))}")


if __name__ == "__main__":
    main()
