# RoadSense dataset utilization

The per-image manifests, gallery, weights, and run outputs referenced below are local research artifacts. Obtain the source datasets and access to the review records before reproducing the runs.

## Approved full-data run

The authorized local run trained on 7,137 images and selected checkpoints using 1,251 validation images. On the 776 author-held StreetSurfaceVis images from five German cities, ResNet50 scored 82.09% (637 correct), MobileNetV2 scored 79.12% (614 correct), and EfficientNetB0 scored 76.93% (597 correct). The [training configuration](../training/configs/quality_full.json) records the lower-center crop, head training, fine-tuning, and validation-based selection settings. ResNet50 recall was 89.0% for Good, 57.0% for Moderate, and 66.3% for Poor.

## What the models used

MobileNetV2, ResNet50, and EfficientNetB0 were deliberately trained as a two-capture-date pilot. The [pilot manifest](../work/cambodian-pilot/manifest.csv) was rebuilt byte for byte from the owner-confirmed unified audit. Its SHA-256 is `7748d83dec8efaac3f311c9a98fca82decc5210ca50ec9c9ea89ab8694fcdc8d`. All three [model sidecars](../backend/models/) and [evaluation records](../backend/results/) point to that hash. The three saved model files also match their sidecar and evaluation hashes.

| Partition | EXIF capture date | Image IDs, inclusive | Good | Moderate | Poor | Total |
|---|---|---|---:|---:|---:|---:|
| Train | 2026-11-05 | CAM051–CAM079 | 6 | 20 | 3 | 29 |
| Test | 2026-11-11 | CAM080–CAM100 | 9 | 10 | 2 | 21 |

[`prepare_cambodian_pilot.py`](../training/prepare_cambodian_pilot.py) filtered an explicitly supplied unified CSV to Cambodian records marked self-collected and permission-confirmed. It required reviewed labels, no recorded duplicate group, and EXIF dates consistent with filenames. A hardcoded assertion required exactly 50 photos and two dates. Sorting the dates put the entire earlier date in train and the entire later date in test. There was no random image selection, sample limit, hardcoded input path, or validation partition in the pilot. The training and evaluation CLIs require `--capture-date-pilot` for that manifest. The [pilot config](../training/configs/cambodian_pilot.json) used batch size 8, 12 frozen-backbone head epochs, no fine tuning, seed 42, class weighting, and training-only augmentation.

The ResNet50 and EfficientNetB0 [training logs](../work/cambodian-pilot/) record all 12 epochs. MobileNetV2 has a saved 12-epoch history in its model sidecar and a saved evaluation log. All three evaluation logs report 21 test examples and the pilot use scope. The README and model metadata call it a pilot. The artifacts were installed as the local application models. The separate date gives a session-level comparison, and camera locations remain unverified. The 21 photos have already been inspected in the [pilot error analysis](../investigation/REPORT.md).

| Not selected for pilot training | Count | Verified reason |
|---|---:|---|
| CAM080–CAM100 | 21 | Entire later capture date reserved for pilot testing |
| CAM001–CAM050 | 50 | Online source and unresolved training permissions. CAM006 is Uncertain and CAM023 duplicates CAM022 byte for byte |
| StreetSurfaceVis | 9,122 | Outside the Cambodian pilot filter. The full-data audit also lacked an approved three-class conversion and an intended-use decision |
| RDD_SPLIT | 38,385 | Outside the Cambodian pilot filter. The local export also lacked verified three-class quality labels and documented export provenance |

The [image decision ledger](../work/quality-dataset-2026-11-16/image_decisions.csv) records all 47,607 image IDs, original partitions, pilot assignments and exclusions, earlier full-data gate reasons, and current decisions. The [owner-confirmed training-ready manifest](../integration/baseline/manifests/owner_confirmed/training_ready.csv) had zero records at the time. The pilot supplied a separate entry point for the owner-authorized photos.

## StreetSurfaceVis mapping and intended use

The local v1.0 CSV and five-page labeling guide match the published MD5 checksums. All 9,122 CSV IDs join one-to-one with the audited images. The preparation command rehashes every local StreetSurfaceVis image against the audited SHA-256 values. Only the unpacked images were available locally for validation. [Dataset release and checksums](https://zenodo.org/records/11449977).

| Original smoothness rating | Proposed RoadSense quality | Original count |
|---|---|---:|
| Excellent | Good | 1,670 |
| Good | Good | 3,238 |
| Intermediate | Moderate | 2,610 |
| Bad | Poor | 1,301 |
| Very bad | Poor | 303 |

The [guide](../dataset/streetSurfaceVis/labeling_guide.pdf) defines smoothness and usability by surface type. I inspected all five pages and one dataset image from each of the 18 surface–rating combinations. Examples include near-new asphalt `2892513444383654`, a concrete surface with visible wear `4120569384673828`, visibly damaged concrete `164321238994866`, and a very bad unpaved track `1129573420865468`. The mapping preserves the guide's order. Good can include early wear, Moderate includes maintained but visibly uneven surfaces and the best compacted unpaved paths, and Poor spans rough or damaged surfaces through tracks that require high clearance. RoadSense's three classes need an explicit decision to adopt that smoothness rubric across asphalt, concrete, paving stones, sett, and unpaved surfaces. The source labels can refer to the focal cycleway, footway, or path when one is visible. The provided metadata has no road-use field to separate those images from motor-vehicle roads. The [review gallery](../integration/baseline/gallery/index.html) has additional examples. [Dataset paper and label definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739464/).

All original ratings, contributor names and profiles, image URLs, and license references are kept in the proposed manifest. The dataset is published under CC BY-SA 4.0. Its published terms allow reuse and adaptation subject to the license, with attribution and applicable share-alike obligations when sharing. The authors also point to Mapillary's terms, whose full page required sign-in during this audit. The scope of sharing images, crops, or model artifacts remains unrecorded. The manifest keeps the original license evidence and leaves RoadSense mapping and intended training use pending human review. [CC BY-SA 4.0 legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en), [Mapillary attribution guidance](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data).

The authors recommend learning from the lower center of each image, where they assigned labels. The proposed split uses the existing full-image model input. [Dataset preprocessing guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739464/).

## Proposed partitions

| Split | StreetSurfaceVis | Cambodian | Good | Moderate | Poor | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 7,087 | 50 | 3,691 | 2,154 | 1,292 | 7,137 |
| Validation | 1,251 | 0 | 649 | 375 | 227 | 1,251 |
| Test | 776 | 0 | 583 | 107 | 86 | 776 |
| Total | 9,114 | 50 | 4,923 | 2,636 | 1,605 | 9,164 |

- The proposal preserves every author-held image from five test cities. Validation uses approximately 15% of the remaining StreetSurfaceVis images, grouped before assignment with seed 42. Groups link image coordinates within 400 m, consecutive captures from the same contributor within 60 seconds, and dHash neighbors within each original partition at Hamming distance 4 or less. The CSV has no sequence or road-segment IDs. The grouping uses observed coordinates, times, and candidate similarities as proxies. The nearest measured StreetSurfaceVis pairs are 400.7 m for train–validation, 439.5 m for train–test, and 2,411.5 m for validation–test. [Author split](https://zenodo.org/records/11449977).
- Six same-label exact duplicates were dropped. Both copies of one identical image with conflicting original ratings were excluded. All eight were on the authors' training side. The [pair review](../integration/baseline/reports/street_cross_split_candidates.csv) records the 15 dHash matches across the authors' split. Side-by-side review showed different scenes, consistent with the recorded distances. No exact or dHash match was detected between the 50 authorized Cambodian images and the StreetSurfaceVis pool. Hash similarity and coordinates have finite coverage.
- The 50 owner-authorized, human-labeled Cambodian photos form one train-only group. This includes the 21 formerly held out by date, which have already been analyzed and have no verified independent locations. The German test set measures generalization to other German locations. Cambodia still needs a separate location-verified test collection. The other 50 Cambodian images are excluded for unresolved online permissions. [Recorded owner confirmation](../integration/baseline/reports/owner_confirmation.json).
- All 38,385 RDD_SPLIT images are ineligible for this supervised quality task. The local train, validation, and test directories hold 26,869, 5,758, and 5,758 images. The audit found 26,661 files with damage boxes and 11,724 with empty annotations. An empty file conveys no recorded damage box. The official RDD2022 release describes four object-level damage categories and lists CC BY 4.0. The supplied YOLO export has no authoritative integer-to-class mapping or accompanying provenance or license record tying it to that release. Neither the boxes nor the 450-image review pilot provide whole-image quality ratings. [Official challenge annotations](https://crddc2022.sekilab.global/data/), [Official release metadata](https://api.figshare.com/v2/articles/21431547).

## Reproduce or approve the split

Run from the repository root with the training dependencies installed. Choose an unused output path.

```sh
python -m training.prepare_quality_dataset prepare \
  --unified integration/baseline/manifests/owner_confirmed/unified_metadata.csv \
  --street-csv dataset/streetSurfaceVis/streetSurfaceVis_v1_0.csv \
  --pilot work/cambodian-pilot/manifest.csv \
  --pair-review integration/baseline/reports/street_cross_split_candidates.csv \
  --output work/quality-dataset-rebuild
```

The [proposed split](../work/quality-dataset-2026-11-16/proposed_splits.csv) is reviewable but blocked from training. Human approval covers the exact proposal and intended usage. Record the approver, evidence of mapping and scope approval, and evidence that the stated training use was reviewed, using the proposal SHA-256 in [summary.json](../work/quality-dataset-2026-11-16/summary.json). The approval JSON fields are `reviewed_by`, `mapping_and_surface_scope_approved` (a boolean), `mapping_approval_evidence`, `training_usage_evidence`, and `proposed_manifest_sha256`.

```sh
python -m training.prepare_quality_dataset approve \
  --proposal work/quality-dataset-2026-11-16/proposed_splits.csv \
  --approval /path/to/documented-approval.json \
  --output work/quality-dataset-2026-11-16/approved_splits.csv
```

Use the resulting `approved_splits.csv` directly as `--manifest` for training and evaluation, with new model and result directories. Validation drives checkpoint selection. Evaluate on the author-held test only after model selection. The saved model sidecar records the training-manifest hash and per-partition counts by source.
