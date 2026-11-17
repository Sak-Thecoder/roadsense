# RoadSense ML investigation

## Scope and evidence

This investigation re-ran all three saved models against the exact 21-image held-out pilot manifest, verified each model's artifact and training-manifest hashes, and reproduced the existing confusion matrices. The 29 training images and 21 test images are grouped by two capture dates. Capture locations are unknown. These test images have now been inspected for diagnosis, so preserve an additional independent final test set before choosing a model or tuning preprocessing.

## Main findings

1. **Dataset size, class imbalance and dependence are the strongest supported concerns.** Training has 6 Good, 20 Moderate and 3 Poor photos. The 21-image test set has 9 Good, 10 Moderate and 2 Poor. Several photos were taken seconds apart and show closely related scenes, so counts are not independent road segments. Both held-out Poor photos appear to show the same damaged road feature. Performance on other roads is unknown. The capture-date split separates sessions but cannot establish location independence because coordinates are missing.
2. **ResNet50's Moderate bias is observable but its single cause is not established.** It predicts Moderate for 17/21 photos, including 8/9 Good and 2/2 Poor. Moderate makes up 20/29 of the training set. Training used inverse-frequency class weights (approximately Good 1.61, Moderate 0.48, Poor 3.22), which reduce the direct loss contribution from the majority class but cannot create additional examples or ensure generalization. Dataset imbalance, related frames, backgrounds and the small sample are plausible contributors. There is no evidence that the imbalance alone caused the bias.
3. **Potential label/field-of-view inconsistency needs owner review.** The contact sheets show a damaged road feature in CAM085, marked Good, which resembles the feature in CAM086 and CAM087, marked Poor. CAM078, marked Good, appears to include a broken road edge. These could reflect the intended focal area or other context. Recheck those decisions against a written rubric without silently changing labels.
4. **Input framing likely affects what the model can learn, though causation is unresolved.** EXIF-corrected inputs are directly resized to 224×224 without cropping or padding. Twenty-eight of 29 train images and all 21 test images are 3000×4000 portrait, which become square and geometrically compressed. Fine cracks occupy fewer pixels, but no bounding-box annotation establishes whether a specific defect is lost. Some photos include vehicles, shadows, road markings, or off-road context. Images are not cropped during training or production inference.
5. **Grad-CAM suggests some predictions may rely on context.** ResNet50's predicted-class overlay for CAM080 highlights the lower-right motorcycle region, and the CAM086 overlay emphasizes the upper vehicle/background region more than the damaged feature. CAM083's highest activation is near the upper-left boundary. These are individual relative activation maps, not proof of causal reliance or damage detection.
6. **Training and held-out results differ substantially.** The last augmented training-epoch accuracies are 86.2%, 93.1%, and 96.6% for MobileNetV2, ResNet50, and EfficientNetB0 respectively. Test accuracies are 33.3%, 33.3%, and 52.4%. The training loss is class-weighted and includes stochastic augmentation. No validation set or validation curves were recorded. The gap is consistent with poor generalization or overfitting, but does not quantify overfitting or isolate the cause.

## Reproduced full-image evaluation

Rows are true classes and columns are predicted classes in the order Good, Moderate, Poor. Precision, recall and F1 use zero for classes with no predicted examples.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Confusion matrix |
|---|---:|---:|---:|---:|---|
| MobileNetV2 | 7/21 (33.3%) | 23.4% | 24.1% | 23.5% | [[2,5,2],[5,5,0],[0,2,0]] |
| ResNet50 | 7/21 (33.3%) | 13.7% | 23.3% | 17.3% | [[0,8,1],[1,7,2],[0,2,0]] |
| EfficientNetB0 | 11/21 (52.4%) | 36.5% | 37.8% | 36.7% | [[3,4,2],[2,8,0],[2,0,0]] |

Every model has 0/2 Poor recall. An always-Moderate classifier gets 10/21 (47.6%) on these same examples. With just two Poor images and repeated scenes, these metrics are highly uncertain and not representative population performance.

Misclassification examples include CAM080, labeled Moderate but predicted Good by all three models, and CAM083, labeled Good but predicted Moderate by all three. CAM086 is labeled Poor and predicted Moderate by ResNet50. The per-image CSV contains the original label, each prediction and all class scores for full and crop input. Separate confusion-matrix images, nine Grad-CAM overlays and an HTML error gallery are alongside this report's generated outputs.

## Full-image versus crop diagnostic

The diagnostic crop uses the central 50% of original width and bottom 50% of original height, as described in the StreetSurfaceVis authors' guidance. It is resized to 224×224 and fed to the unchanged models. It may exclude off-center road defects. The models were trained on full-image resize, so the crop is an out-of-distribution intervention and is not a fair assessment of a crop-trained model.

| Model | Full image | Diagnostic crop | Full predicted Poor | Crop predicted Poor |
|---|---:|---:|---:|---:|
| MobileNetV2 | 7/21 (33.3%) | 6/21 (28.6%) | 2 | 11 |
| ResNet50 | 7/21 (33.3%) | 5/21 (23.8%) | 3 | 14 |
| EfficientNetB0 | 11/21 (52.4%) | 7/21 (33.3%) | 2 | 12 |

The crop classified both Poor photos correctly for each model, but increased Poor predictions sharply and reduced overall accuracy. These are exploratory test-set observations, not evidence to deploy or choose the crop. Design future crop and padding comparisons using an independent validation set.

## Preprocessing and class order audit

Training maps labels through `CLASSES = (Good, Moderate, Poor)`. Artifact sidecars use the same order, the registry rejects incompatible sidecars, evaluation uses that order for matrices, and API output maps softmax indices through the shared tuple. All model hashes match their sidecars and their training manifest. The shared `decode` and `preprocess` functions are used for training, evaluation and API inference. Existing numerical tests compared all three transforms against the installed Keras applications. MobileNetV2 uses [-1,1] scaling, ResNet50 uses RGB-to-BGR and ImageNet mean subtraction, and EfficientNetB0 receives 0–255 pixels with its rescaling inside the saved backbone. EXIF orientation is handled. There is no supported preprocessing or class-order bug to fix from the available evidence.

The three saved architectures use a frozen ImageNet backbone, global average pooling, 25% dropout, and a three-class softmax head. They were trained for 12 fixed epochs, with horizontal flip and small rotation only during training. No fine-tuning or validation-based selection occurred. Final training losses were 0.306, 0.254 and 0.374 respectively. These weighted training losses are not comparable to an unavailable validation loss.

## Data sources and proposed mapping

- Cambodian confirmed labels total 39 Good, 40 Moderate, 20 Poor and 1 Uncertain. Only the 50 self-collected photos were used in this pilot. Their owner confirmed training permission. The 50 online-image permissions remain unconfirmed.
- StreetSurfaceVis contains 1,670 excellent, 3,238 good, 2,610 intermediate, 1,301 bad and 303 very_bad original labels. The proposed mapping gives 4,908 Good, 2,610 Moderate and 1,604 Poor. It is a plausible ordinal aggregation, not an approved or empirically validated match to RoadSense's full-image criteria. Surface type, focal region and whether paths belong in scope can affect comparability. The authors' guide and review gallery remain available in `integration/baseline/gallery`. Preserve the five-level labels and geographic test flags.
- RDD_SPLIT damage classes are object categories, and the supplied YOLO ID mapping and full-image quality labels remain unresolved. Its 450-image pilot is unlabelled for road quality.
- The previous cross-source audit found nine extra exact duplicates, 873 perceptual components, and 15 direct StreetSurfaceVis train/test dHash candidates. These are candidates rather than verified overlap. See `integration/baseline/reports/DUPLICATES_AND_SPLITS.md`.

Visual contact sheets show many sequential frames with recurring road textures, objects, shadows and markings. The training set contains several shots of the same concrete seam and of the same shoulder. The test set includes repeated shots of a broken feature and crosswalks. Across the two splits, median thumbnail luminance is approximately 153.7 and 153.6 on a 0–255 scale, with median edge-filter means 23.7 and 22.9. These simple summaries do not rule out differences in scene, damage, viewpoint, lighting, or location.

## Prioritized next steps

1. Define a focal road region and review rubric, then independently review potential label inconsistencies. Do not relabel automatically.
2. Collect substantially more independent, rights-cleared Cambodian road segments and examples of Poor conditions. Record location/session groups and include multiple devices, surfaces and lighting conditions.
3. Assign disjoint scene or geographic groups to train, validation and a new final test set. Keep the current 21-image set as an exploratory diagnostic set. Resolve cross-source and near-duplicate candidates.
4. Evaluate the StreetSurfaceVis mapping on the review gallery, approve scope and rights separately, preserve its author geographic test partition, and use its training regions only when approved.
5. On training/validation only, compare aspect-preserving resize or padding and agreed road-focused crops, class balancing, and suitable augmentation. With enough data, compare a frozen backbone against careful fine-tuning. Use a fixed protocol and identical validation partitions across models.
6. Evaluate the final chosen model once on a new independent test set, reporting per-class metrics and uncertainty.

## Reproduce

From the repository root, install `training/requirements.txt`, then run:

```sh
TF_CPP_MIN_LOG_LEVEL=2 .venv/bin/python -m investigation.diagnose \
  --manifest work/cambodian-pilot/manifest.csv --out work/ml-investigation-reproduction
.venv/bin/python -m pytest backend/tests/test_api.py::test_preprocess_matches_keras \
  backend/tests/test_api.py::test_exif_orientation \
  backend/tests/test_training.py::test_capture_date_pilot_requires_explicit_mode_and_keeps_test_separate -q
```

The output directory must not already exist. The diagnostic script verifies saved model metadata, source image hashes, and existing full-image confusion matrices before writing its report outputs. No training or label changes are performed.
