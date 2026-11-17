# RoadSense baseline preparation

Start with [STATUS.md](reports/STATUS.md), [RIGHTS.md](reports/RIGHTS.md), and [DUPLICATES_AND_SPLITS.md](reports/DUPLICATES_AND_SPLITS.md). The datasets, gallery, manifests, and per-image audit outputs are local research files. Open `gallery/index.html` locally once the gallery has been generated. It is a stratified review aid, not a random estimate of dataset prevalence. Gallery previews are derived from the original images and carry the source license; do not redistribute without appropriate rights and attribution.

The owner-confirmed current records are `manifests/owner_confirmed/cambodian_reviewed.csv` and `manifests/owner_confirmed/unified_metadata.csv`. The original source records and previous integration audit are unchanged. The 95 existing entries were explicitly confirmed, not inferred from folders. An owner attestation is recorded at `reports/owner_confirmation.json`; the reviewer name and inspection date are blank because they were not supplied. The 50 online images remain unconfirmed for training. The RDD pilot remains at `../../work/full/pilot.csv`.

To correct recorded values or add known reviewer/date/group/rights information without viewing the images again:

```sh
pip install -r requirements.txt
streamlit run integration/baseline/scripts/review_app.py
```

The table starts with all existing decisions. Edit cells and save a new version; it does not alter the original or the current unified manifest. Inspection date must be supplied as `YYYY-MM-DD`, and should remain blank if unknown. For online images, retain `unconfirmed` unless you have evidence; record that evidence in `rights_notes`. To apply a saved version to unified metadata, provide its actual filename and a new output directory:

```sh
python integration/baseline/scripts/apply_review.py \
  --review integration/baseline/manifests/review_versions/REPLACE_WITH_SAVED_FILENAME.csv \
  --unified integration/baseline/manifests/owner_confirmed/unified_metadata.csv \
  --output integration/baseline/manifests/next_version
```

`apply_review.py` validates IDs and labels, requires an ISO review date when present, requires notes for confirmed online permissions, and writes a new unified and training-ready CSV. It will not overwrite the output directory. The current eligibility gate also requires a documented reviewer/date, verified capture group, accepted label, confirmed training permission, and no unresolved duplicate candidate. CAM023 remains excluded as an exact copy; neither original is deleted. This gate is deliberately conservative; it does not establish that a split is safe or that a user's permission claim is legally sufficient. A corrected review version should be checked for appropriate reviewer, group, and rights evidence before use.

The two original Cambodian folder categories are `damage_road` and `moderate_to_good_road`, plus a `self_collected_road` folder; these are not three-class folders and cannot be automatically imported as verified labels. All 100 recorded labels have now been confirmed by the owner, so no reinspection is needed merely to synchronize status. Changes can be made individually in the table.

For validation:

```sh
python -m unittest discover -s integration/baseline/tests -v
python -m unittest discover -s integration/tests -v
python integration/baseline/scripts/geographic_check.py --input integration/baseline/manifests/owner_confirmed/unified_metadata.csv --output /tmp/roadsense_geographic_check.csv
```

Use a new path for the geographic output. `prepare_baseline.py` describes how the initial gallery and proposal were generated and refuses to overwrite its existing main outputs. `sync_cambodian.py` only accepts the exact 95-label snapshot explicitly approved by the owner and creates a new output directory. The gallery, provisional mapping and permissions evidence are not approvals to train. Future validation must use the author test partition exclusively as a test set and create a separate group-aware validation set from training data.
