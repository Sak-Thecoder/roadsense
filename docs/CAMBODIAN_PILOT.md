# Cambodian capture-date pilot

RoadSense has three locally installed pilot models trained from the owner-authorized, reviewed Cambodian photos. The pilot manifest uses EXIF capture dates checked against the filenames. The 50 photos have unique file hashes and no recorded perceptual duplicate group. Their locations could not be verified from the available metadata.

| Capture date | Use | Good | Moderate | Poor | Total |
|---|---|---:|---:|---:|---:|
| 2026-11-05 | Train | 6 | 20 | 3 | 29 |
| 2026-11-11 | Held-out test | 9 | 10 | 2 | 21 |

All three runs used the same fixed 12-epoch schedule. Each trained a three-class head on a frozen ImageNet backbone with class weighting and training-only augmentation. The test date was reserved for evaluation. The two dates leave no third capture group for validation.

| Model | Correct | Accuracy | Macro F1 | Median CPU inference |
|---|---:|---:|---:|---:|
| MobileNetV2 | 7/21 | 33.3% | 0.235 | 56.1 ms |
| ResNet50 | 7/21 | 33.3% | 0.173 | 136.0 ms |
| EfficientNetB0 | 11/21 | 52.4% | 0.367 | 119.9 ms |

Predicting the most frequent training class, Moderate, for every test photo would give 10/21 correct (47.6%). All three models missed both Poor examples. These results support a functional application demonstration and reveal the need for a larger geographically grouped dataset before relying on classifications. Road locations may overlap across dates. The benchmark applies no geographic independence claim.

Latency is the median of 100 synchronous batch-one measurements after 10 warm-ups on the local ARM Mac CPU with TensorFlow 2.21.0. It includes model-specific pixel preprocessing and excludes image decoding, model loading, and network time. The per-model JSON, confusion matrices, and metric charts are in `backend/results/`. The original project-wide approval manifest still tracks the evidence needed for the broader study. See the [README](../README.md) for pilot reproduction commands.
