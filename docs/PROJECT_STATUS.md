# RoadSense project status before implementation

The checkout contains the Python dataset audit, RDD pilot, Streamlit labeling tools, and the three-source integration reports. The original datasets are held outside this repository. The owner-confirmed manifest covers 47,607 images and its training-ready CSV has zero records.

| Source | Images | Baseline status |
|---|---:|---|
| RDD_SPLIT | 38,385 | 450-image review pilot. No overall quality labels and no authoritative mapping for this YOLO export. |
| StreetSurfaceVis | 9,122 | Proposed three-class mapping awaits approval. Dataset license evidence and the original 776-image geographic test set are recorded in the baseline report. |
| Cambodian | 100 | 39 Good, 40 Moderate, 20 Poor, 1 Uncertain. The owner authorized 50 self-collected images for RoadSense training. Their capture groups remain unresolved and the online images lack confirmed training rights. |

No trained Keras files or measured benchmark results were found. Training and evaluation require a reviewed and leakage-checked split manifest. The application will read real model and evaluation artifacts when those become available. See [dataset baseline status](../integration/baseline/reports/STATUS.md).
