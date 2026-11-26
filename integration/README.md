# RoadSense dataset integration

> **Audit record from 2026-11-15.** It describes the data before training. Since then the StreetSurfaceVis mapping and split were approved (2026-11-16) and the quality_v3 models were trained on them. See [docs/DATASET_UTILIZATION.md](../docs/DATASET_UTILIZATION.md) for how the data is used now.

This directory audits the available three datasets without changing their originals. It does not train a model or assign train, validation, or test splits. The dated audit folder contains the results; `sample_run` is an initial 30-per-source test, not a representative sample.

## Current outputs

- `audit_2026-11-15/unified_metadata.csv`: one record per original image, including the original label, a proposed conversion where applicable, a separate verified three-class label, rights status, source paths, duplicate group, and exclusion reasons. Cambodian originals remain inside their original ZIP: `image_path` is the archive and `archive_member` is the exact member name. StreetSurfaceVis source CSV values and Cambodian prior manifest values are preserved as JSON in `source_metadata`; RDD annotations are linked by path.
- `audit_2026-11-15/training_ready.csv`: rows satisfying all present eligibility checks. It currently has a header only; no images have both an accepted three-class label and recorded confirmed training permission. Its name does not imply that future rows have safe train/test group assignments.
- `audit_2026-11-15/duplicate_candidates.csv`: members of perceptual similarity components, including hashes and original split flags. Review similarity before deciding which images are duplicates or related scenes.
- `audit_2026-11-15/issues.csv` and `summary.json`: file errors and counts. Read the report for limitations.

`label_status=reviewed` for a Cambodian image reflects the source manifest, not an independent re-review; the five reviewed records have blank reviewer and review date. Its other 95 label values were marked unreviewed and are preserved only as original proposals. Folder names are never assigned as quality labels. The existing `work/full/pilot.csv` from the prior RDD audit remains available for manual labeling; no completed RDD pilot label file was found.

## Proposed StreetSurfaceVis conversion

| Original | Proposed | Rationale / review |
|---|---|---|
| excellent, good | Good | Combines new and lightly worn; decide whether lightly worn `good` matches RoadSense criteria. |
| intermediate | Moderate | Covers repaired, cracked or uneven examples; review the Good/Moderate and Moderate/Poor boundaries. |
| bad, very_bad | Poor | Combines two severity levels; retain the original value to distinguish them. |

This is a **proposal**, not an accepted label conversion. The supplied guide defines quality differently by surface material and includes unpaved and narrow paths: the same original rating can represent a different surface and usage context. Decide if unpaved roads, paving stones, sett, sidewalks and paths belong in the intended model scope; review borderline examples and adopt a written labeling policy before accepting the conversion. Neither `bad` damage nor an empty RDD box file automatically determines a quality class. The original StreetSurfaceVis CSV has `train=True/False`; the latter is recorded as `non_train` because the supplied files do not establish its test/validation role. Coordinate and contributor values are retained, not interpreted as permission or sequence IDs.

## Repeat an audit

From the repository root, with Python 3.10+ and `pip install -r requirements.txt`:

```sh
python integration/scripts/audit_integration.py --dataset-root /path/to/dataset --output integration/sample_new --limit-per-source 30
python integration/scripts/audit_integration.py --dataset-root /path/to/dataset --output integration/audit_new --workers 8
python -m unittest discover -s integration/tests -v
```

The output path must not already exist. `--dry-run` writes nothing. Paths and image hashes are retained, originals are not modified, and ZIP members are decoded in memory. The script reads the three fixed dataset folder names and the layout observed here; investigate new layouts before running it. Similarity grouping uses the existing `roadsense.audit.group_duplicates` implementation: dHash Hamming distance <=4, then transitive closure. False positives and negatives are expected. Large groups are especially prone to chaining; inspect candidate relationships and use independently verified segment/sequence/capture IDs. Source identifiers, surface, original labels, rights fields, and split flags are not interchangeable.

## Before training

1. Obtain verifiable training rights and attribution requirements for every image, including the 50 online and 50 self-collected Cambodian photos (the supplied records mark both unconfirmed), StreetSurfaceVis, and RDD images. A license field or online availability alone is not proof of permission. Record evidence and retain source terms.
2. Review Cambodian records and document reviewer/date; verify location and capture groups. The five marked reviewed have no reviewer/date in the provided export. Label RDD images manually if using them for overall road quality. Approve and validate the proposed StreetSurfaceVis conversion; do not promote it merely because it is populated.
3. Review exact and near duplicates, cross-source overlaps, and related scenes; record grouping metadata, including Mapillary sequence IDs where available. Existing split flags do not establish independence. Do not treat a hash group as a confirmed duplicate.
4. After rights, label, duplicate, and grouping review, create new versioned manifests, assess class/source balance and domain differences, and make group-aware train/validation/test splits. Keep any future augmentation exclusively in training. Reserve an independent Cambodian-only evaluation set once enough rights-cleared, location-verified, independently grouped and labeled local images exist.

The script is conservative: only a reviewed `Good`/`Moderate`/`Poor` label, a permission status of `confirmed`, `granted`, or `approved`, a readable image, and no unresolved duplicate candidate qualify. It does not verify legal evidence or guarantee grouping; users must check these separately. `eligible` is not a claim that a model is ready to train. Change source records only through a separate, auditable review process; rerun to a new output directory.
