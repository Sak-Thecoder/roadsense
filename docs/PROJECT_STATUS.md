# RoadSense project status

Last updated 2026-11-27.

## Done

- **Dataset:** StreetSurfaceVis (9,122 images) audited, mapped from five smoothness levels to Good / Moderate / Poor, and split with location, time and near-duplicate grouping. The split and mapping were approved on 2026-11-16. It has 7,137 training, 1,251 validation and 776 test images. The 50 owner-authorized Cambodian photos are in the training set. See [DATASET_UTILIZATION.md](DATASET_UTILIZATION.md).
- **Models:** MobileNetV2, ResNet50 and EfficientNetB0 trained with `training/configs/quality_v3.json` (surface-aware fine labels, label smoothing, full fine-tuning, class weights, selection by validation macro F1).
- **Test results** (776 held-out images, evaluated once):

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| MobileNetV2 | 82.7% | 69.6% |
| ResNet50 | 84.1% | 73.2% |
| EfficientNetB0 | 80.8% | 70.2% |

- **App:** FastAPI backend with checksum-verified models and a SegFormer road check. React front end with capture, upload, model comparison, benchmark and scan history.
- **Tests:** backend pytest suite, frontend unit tests and Playwright end-to-end tests.

## Not done yet

- **Cambodian accuracy is unknown.** All test images are German. The earlier 21-photo pilot ([CAMBODIAN_PILOT.md](CAMBODIAN_PILOT.md)) is too small, and those photos are now used for training. A location-verified Cambodian test set is needed.
- **Moderate is weak.** F1 is about 44–47% for every model, so the 80% macro F1 goal was not reached.
- **Other data is unused.** The 50 online Cambodian images lack confirmed training permission. RDD_SPLIT (38,385 images) has damage boxes but no whole-image quality labels or class mapping.
