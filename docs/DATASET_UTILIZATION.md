# RoadSense dataset utilization

The per-image manifests, gallery, weights, and run outputs referenced below are local research artifacts. Obtain the source datasets and access to the review records before reproducing the runs.

## Current models (quality_v3)

The models served by the API were trained on 2026-11-19 with [`quality_v3.json`](../training/configs/quality_v3.json) on the approved split described below: 7,137 training images (7,087 StreetSurfaceVis + 50 owner-authorized Cambodian photos) and 1,251 validation images. The best epoch was chosen by validation macro F1. The 776 author-held StreetSurfaceVis test images from five German cities were evaluated once, after all tuning. The training manifest SHA-256 is `094bc9d10c3933b40712917aba4d8e773d6b177dd29a2fed2050ee66aa9d9c35`, and every model sidecar and evaluation record in `backend/` points to it.

| Model | Correct | Accuracy | Macro F1 | F1 Good | F1 Moderate | F1 Poor |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV2 | 642 / 776 | 82.7% | 69.6% | 91.5% | 44.8% | 72.6% |
| ResNet50 | 653 / 776 | 84.1% | 73.2% | 91.6% | 47.1% | 81.0% |
| EfficientNetB0 | 627 / 776 | 80.8% | 70.2% | 89.8% | 44.4% | 76.4% |

About 75% of the test images are Good, so macro F1 is the fairer summary. Moderate is the weakest class for every model. All test images are German, so these numbers do not establish accuracy on Cambodian roads.

## Version history

All three versions used the same approved split and test set.

| Version | Date | Main changes | MobileNetV2 | ResNet50 | EfficientNetB0 |
|---|---|---|---:|---:|---:|
| v1 ([`quality_full.json`](../training/configs/quality_full.json)) | 2026-11-16 | Three-class head, lower-centre crop, head training then fine-tuning | 79.1% / 68.0% | 82.1% / 72.1% | 76.9% / 67.6% |
| v2 ([`quality_v2.json`](../training/configs/quality_v2.json)) | 2026-11-18 | Full-resolution focal crop, balanced class weights, stronger augmentation, selection by validation macro F1 | 77.1% / 67.2% | 81.3% / 71.0% | 76.4% / 66.4% |
| v3 ([`quality_v3.json`](../training/configs/quality_v3.json)) | 2026-11-19 | Surface type × level fine labels summed back to 3 grades, label smoothing 0.1, full fine-tuning with BatchNorm frozen | **82.7% / 69.6%** | **84.1% / 73.2%** | **80.8% / 70.2%** |

Each cell is test accuracy / macro F1.

## Earlier Cambodian pilot (replaced)

Before the StreetSurfaceVis split was approved, the three architectures were trained as a two-capture-date pilot on the 50 owner-authorized Cambodian photos only. The earlier date (2026-11-05, CAM051–CAM079, 29 photos) was used for training and the later date (2026-11-11, CAM080–CAM100, 21 photos) for testing. [`prepare_cambodian_pilot.py`](../training/prepare_cambodian_pilot.py) built that manifest (SHA-256 `7748d83dec8efaac3f311c9a98fca82decc5210ca50ec9c9ea89ab8694fcdc8d`), and [`cambodian_pilot.json`](../training/configs/cambodian_pilot.json) trained a head on a frozen backbone for 12 epochs. Test accuracy was 33.3% to 52.4%; see [CAMBODIAN_PILOT.md](CAMBODIAN_PILOT.md) and the [pilot error analysis](../investigation/REPORT.md). Those models were replaced by the full-data models. The 50 photos are now part of the training set, so no independent Cambodian test set exists yet.

The [image decision ledger](../work/quality-dataset-2026-11-16/image_decisions.csv) records all 47,607 audited image IDs, their original partitions and the decision for each.

## StreetSurfaceVis mapping and intended use

The local v1.0 CSV and five-page labeling guide match the published MD5 checksums. All 9,122 CSV IDs join one-to-one with the audited images. The preparation command rehashes every local StreetSurfaceVis image against the audited SHA-256 values. Only the unpacked images were available locally for validation. [Dataset release and checksums](https://zenodo.org/records/11449977).

| Original smoothness rating | RoadSense class | Original count |
|---|---|---:|
| Excellent | Good | 1,670 |
| Good | Good | 3,238 |
| Intermediate | Moderate | 2,610 |
| Bad | Poor | 1,301 |
| Very bad | Poor | 303 |

The [guide](../dataset/streetSurfaceVis/labeling_guide.pdf) defines smoothness and usability by surface type. I inspected all five pages and one dataset image from each of the 18 surface–rating combinations. Examples include near-new asphalt `2892513444383654`, a concrete surface with visible wear `4120569384673828`, visibly damaged concrete `164321238994866`, and a very bad unpaved track `1129573420865468`. The mapping preserves the guide's order. Good can include early wear, Moderate includes maintained but visibly uneven surfaces and the best compacted unpaved paths, and Poor spans rough or damaged surfaces through tracks that require high clearance. RoadSense's three classes need an explicit decision to adopt that smoothness rubric across asphalt, concrete, paving stones, sett, and unpaved surfaces. The source labels can refer to the focal cycleway, footway, or path when one is visible. The provided metadata has no road-use field to separate those images from motor-vehicle roads. The [review gallery](../integration/baseline/gallery/index.html) has additional examples. [Dataset paper and label definitions](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739464/).

All original ratings, contributor names and profiles, image URLs, and license references are kept in the split manifest. The dataset is published under CC BY-SA 4.0. Its published terms allow reuse and adaptation subject to the license, with attribution and applicable share-alike obligations when sharing. The authors also point to Mapillary's terms, whose full page required sign-in during this audit. The scope of sharing images, crops, or model artifacts remains unrecorded. The manifest keeps the original license evidence. The mapping above and the intended training use were approved on 2026-11-16 (approval record: `work/quality-retrain-20261116-012646/approval.json`). [CC BY-SA 4.0 legal text](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en), [Mapillary attribution guidance](https://help.mapillary.com/hc/en-us/articles/115001770409-CC-BY-SA-license-for-open-data).

The authors recommend learning from the lower center of each image, where they assigned labels. The current models follow this: each image is cropped to its lower-centre road region (central half of the width, lower half of the height) at full resolution before resizing to 224×224, and the API applies the same crop. [Dataset preprocessing guidance](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739464/).

## Partitions

| Split | StreetSurfaceVis | Cambodian | Good | Moderate | Poor | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 7,087 | 50 | 3,691 | 2,154 | 1,292 | 7,137 |
| Validation | 1,251 | 0 | 649 | 375 | 227 | 1,251 |
| Test | 776 | 0 | 583 | 107 | 86 | 776 |
| Total | 9,114 | 50 | 4,923 | 2,636 | 1,605 | 9,164 |

- The split preserves every author-held image from five test cities. Validation uses approximately 15% of the remaining StreetSurfaceVis images, grouped before assignment with seed 42. Groups link image coordinates within 400 m, consecutive captures from the same contributor within 60 seconds, and dHash neighbors within each original partition at Hamming distance 4 or less. The CSV has no sequence or road-segment IDs. The grouping uses observed coordinates, times, and candidate similarities as proxies. The nearest measured StreetSurfaceVis pairs are 400.7 m for train–validation, 439.5 m for train–test, and 2,411.5 m for validation–test. [Author split](https://zenodo.org/records/11449977).
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

The [proposed split](../work/quality-dataset-2026-11-16/proposed_splits.csv) was approved on 2026-11-16, and the resulting `approved_splits.csv` is the manifest used for v1, v2 and v3. To repeat the approval for a rebuilt proposal: human approval covers the exact proposal and intended usage. Record the approver, evidence of mapping and scope approval, and evidence that the stated training use was reviewed, using the proposal SHA-256 in [summary.json](../work/quality-dataset-2026-11-16/summary.json). The approval JSON fields are `reviewed_by`, `mapping_and_surface_scope_approved` (a boolean), `mapping_approval_evidence`, `training_usage_evidence`, and `proposed_manifest_sha256`.

```sh
python -m training.prepare_quality_dataset approve \
  --proposal work/quality-dataset-2026-11-16/proposed_splits.csv \
  --approval /path/to/documented-approval.json \
  --output work/quality-dataset-2026-11-16/approved_splits.csv
```

Use the resulting `approved_splits.csv` directly as `--manifest` for training and evaluation, with new model and result directories. Validation drives checkpoint selection. Evaluate on the author-held test only after model selection. The saved model sidecar records the training-manifest hash and per-partition counts by source.
