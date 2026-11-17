# RoadSense · Road Surface Quality Detection

RoadSense is an AUPP project for classifying the visible quality of a road surface as **Good**, **Moderate**, or **Poor**. The React app supports camera capture, image upload, model comparison, benchmark results, and local scan history. It calls FastAPI for real inference using available trained Keras artifacts. See the [implementation starting point](docs/PROJECT_STATUS.md) and [dataset baseline](integration/baseline/reports/STATUS.md).

## Run locally

The repository contains the source, test fixtures, and training tools. Place authorized datasets and research manifests in the local paths shown in the training instructions. To enable inference and benchmark results in a fresh clone, supply trained model files with their metadata sidecars and evaluation results, or generate those artifacts using the training and evaluation commands below. The Playwright suite supplies its own fixture models.

Use Python 3.11 and Node.js 22.12+ or Node.js 24. Install dependencies and start FastAPI in the first terminal.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m backend.app.road_photo
python -m uvicorn backend.app.main:app --reload
```

From the repository root in the second terminal:

```sh
npm --prefix frontend ci
npm --prefix frontend run dev
```

Open `http://localhost:5173`. FastAPI serves OpenAPI at `http://localhost:8000/docs`. The Vite development proxy forwards `/api` to port 8000. Camera access requires localhost or HTTPS and browser/device permission. Upload accepts JPEG, PNG, and WebP. Analyze and Compare require visible road surface near the lower center of the frame. Photos without sufficient road evidence return an invalid-road-photo error and produce no scan history entry. Scan history stores prediction metadata in the browser.

The REST API exposes `GET /health`, `GET /models`, `POST /predict/{model_name}`, `POST /compare`, `GET /benchmark`, and `GET /benchmark/{model_name}`. Use the multipart field `image`. Available model IDs are `mobilenetv2`, `resnet50`, and `efficientnetb0`. The model selector shows the artifacts available to the running API. The local Cambodian pilot is identified in the scan and benchmark views.

## Approved data, training, and evaluation

Use a CSV with the unified metadata schema, approved three-class labels, confirmed permission, verified scene groups, and hashes. Existing audit and review workflows are below. `prepare_manifest` preserves author test partitions, groups scenes and exact/perceptual duplicate groups, and uses a seeded approximate 70/15/15 split when there is no existing test partition. It attempts a Cambodian holdout when verified local groups are sufficient and both sides retain all three classes. Inspect the resulting per-class counts before committing to a split.

```sh
python -m pip install -r training/requirements.txt
python -m training.prepare_manifest --manifest /path/to/approved.csv --output work/approved_splits.csv
for architecture in mobilenetv2 resnet50 efficientnetb0; do
  python -m training.train --manifest work/approved_splits.csv --model "$architecture" \
    --config training/configs/quality_full.json --output-dir work/quality-run/models
  python -m training.evaluate --manifest work/approved_splits.csv --model "$architecture" \
    --model-dir work/quality-run/models --output-dir work/quality-run/results
done
```

The training input pipeline verifies image hashes, reads ZIP members directly, applies augmentation only to train, and shares decoder and preprocessing code with the API. Training uses ImageNet initialization, a frozen classifier stage, and optional selective fine tuning. Set `ROADSENSE_MODELS_DIR` and `ROADSENSE_RESULTS_DIR` to the output directories when starting the API. They default to `backend/models` and `backend/results`. Evaluation measures held-out metrics, inference latency, and artifact size and saves JSON and PNGs. Use a separate output directory when evaluating `cambodia_holdout` to retain both sets of results.

The [dataset utilization audit](docs/DATASET_UTILIZATION.md) documents the proposed StreetSurfaceVis-led split, its review and approval steps, and the per-image decisions. Use its approved split directly once the StreetSurfaceVis mapping, surface scope and training usage are documented. The 50 owner-authorized Cambodian images are grouped in training. An independent Cambodian benchmark requires location-verified groups.

### Local Cambodian pilot

The pilot uses only the 50 owner-authorized Cambodian photos with confirmed quality labels. Capture dates come from EXIF and agree with the filenames. The earlier date contributes 29 photos to training. The later date contributes 21 photos to testing. The fixed 12-epoch schedule trains the classification head on a frozen ImageNet backbone, with class weighting based on the training date. No photo from the later date is used for training or tuning. Capture locations remain unverified, so this test measures performance on a separate capture date. It does not establish accuracy on unseen locations.

To reproduce the pilot in a fresh output directory:

```sh
python -m training.prepare_cambodian_pilot \
  --manifest integration/baseline/manifests/owner_confirmed/unified_metadata.csv \
  --output work/pilot-rebuild/manifest.csv
for architecture in mobilenetv2 resnet50 efficientnetb0; do
  python -m training.train --capture-date-pilot --manifest work/pilot-rebuild/manifest.csv \
    --config training/configs/cambodian_pilot.json --model "$architecture" --output-dir work/pilot-rebuild/models
  python -m training.evaluate --capture-date-pilot --manifest work/pilot-rebuild/manifest.csv \
    --model "$architecture" --model-dir work/pilot-rebuild/models --output-dir work/pilot-rebuild/results
done
```

## Verification and deployment

```sh
. .venv/bin/activate
python -m pip install -r backend/requirements-dev.txt -r training/requirements.txt
python -m pytest backend/tests -q
npm --prefix frontend run test
(cd frontend && npx playwright install chromium)
npm --prefix frontend run test:e2e
npm --prefix frontend run build
```

The browser suite starts an isolated FastAPI service with small test Keras models and evaluates them on test fixtures. The two historical dataset assertions under `integration/baseline/tests` require the local research CSVs and skip when those inputs are absent.

See [deployment](docs/DEPLOYMENT.md) for Docker Compose, Vercel, and a hosted FastAPI service.

---

# RDD_SPLIT audit and human review

The currently available dataset is an extracted `RDD_SPLIT` tree, not the original Pascal VOC distribution: it has `train`, `val`, and `test` image and YOLO text-label folders. No class-ID mapping, sequence IDs, or license records were found. Numeric IDs remain numeric: do not assume they correspond to D00/D10/D20/D40. The audit also recognizes adjacent Pascal VOC XML or XML under a sibling `annotations` directory and reports its object names verbatim. Other layouts require adaptation; verify annotation matching before relying on output.

Create a Python environment and install `requirements.txt`. Run from this repository:

```sh
python -m roadsense.audit --root /path/to/RDD_SPLIT --out work/sample --limit 100 --pilot-size 60
python -m roadsense.audit --root /path/to/RDD_SPLIT --out work/full --pilot-size 450 --seed 42
streamlit run roadsense/streamlit_app.py
```

The sample is sorted and not representative. The full audit reads and decodes all discovered images locally and writes `metadata.csv`, `pilot.csv`, `issues.csv`, `summary.json`, `REPORT.md`, and `audit_charts.png`. The metadata includes image dimensions, annotation counts, hashes, an empty human label, duplicate component, and blank scene/provenance fields. `country` is inferred from the filename prefix in this dataset, and China capture types are left separate. Check those values against original source records. Images and annotations are not copied or modified. `--dry-run` calculates without writing output. Run the sample before the full dataset.

The pilot uses a deterministic round-robin over country/prefix and recorded classes, including a no-recorded-boxes stratum, and selects at most one image per detected duplicate component. Its distribution is not a representative sample of the population. Difference hash at distance <= 4 detects some similar images, but misses other related scenes and can merge visually distinct images. Supply verified `scene_group` IDs when available and review candidates manually; filenames do not establish capture sessions. Keep license, attribution, and source URLs in the corresponding metadata columns for any additional Mapillary or locally collected images. Do not scrape or redistribute images without authorization.

The Streamlit app opens `work/full/pilot.csv` by default. Set paths in its sidebar if needed. Select Good, Moderate, Poor, Uncertain, or Reject based on visual review; buttons save a separate `pilot_labels.csv` immediately. Use Previous/Next to revisit and correct decisions. The sidebar reports the number reviewed and label counts. Empty annotations and damage IDs are not quality labels. Use one labeling session at a time and back up the progress file.

After review, prepare a split without training:

```sh
python -m roadsense.prepare --metadata work/full/pilot.csv --labels work/full/pilot_labels.csv --output work/full/pilot_splits.csv --minimum-per-class 50 --seed 42 --dry-run
python -m roadsense.prepare --metadata work/full/pilot.csv --labels work/full/pilot_labels.csv --output work/full/pilot_splits.csv --minimum-per-class 50 --seed 42
```

Only reviewed Good/Moderate/Poor rows are marked `eligible=True`; Uncertain, Reject, and unlabeled images remain assigned to protect group boundaries. Review class and split distributions, and use only training data for any future augmentation. Targets are approximate and not class-stratified. `--minimum-per-class` is a user-adjustable shortage threshold, not a statistical guarantee. Splitting joins duplicate and supplied scene groups transitively; without verified scene IDs or complete duplicate detection it cannot guarantee no leakage. The code reserves a Cambodian-only holdout only when at least ten independent, explicitly Cambodian components contain usable human labels. This minimum is a conservative trigger, not evidence of a representative evaluation set; no such imagery is in the current data. Do not use the holdout for training or model selection. Seed, metadata order, and group definitions affect the result. No augmentation or model training is performed here.

Run `python -m unittest discover -s tests -v`. The older general-purpose CLI below remains available; its CSV schema differs from the RDD audit workflow above.

---

# RoadSense dataset preparation

This repository starts with preparation only. It does not train a model, download images, or redistribute them. No datasets were present when this pipeline was built; adapt paths and verify a sample against the actual datasets before a full run. Python 3.10+ and Pillow are required. The labeling UI also requires Python's Tk support.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Inputs and provenance

Keep source images and XML in their original locations. Create a UTF-8 CSV manifest per source with header:

```csv
path,group_id,country,quality_label,license,attribution,source_url
city/001.jpg,session-001,Cambodia,,license identifier,photographer or provider,https://example.org/source
```

`path` is relative to that source's image root (or an absolute path). Use stable road-segment or capture-session identifiers, or the actual Mapillary sequence ID. IDs are scoped by source, so use the same source name for all parts of a dataset; overlapping segments across sources need manual reconciliation. Set `country` explicitly to `Cambodia` or `KH` for eligible local holdout images, including Cambodian Mapillary images. Blank means unknown, not Cambodian. Record the actual applicable license, attribution and source URL; blanks mean unknown, not permission granted. Keep original source records and license files separately. The manifest may have blank `quality_label`, or a previously human-assigned `Good`, `Moderate`, `Poor`, or `Uncertain` value. XML damage classes are recorded separately and never converted into quality labels. Review image rights and provenance before use.

Each XML path must mirror the image path under `--annotations` with an `.xml` extension; alternatively omit that option for adjacent XML. For example, `images/Japan/0001.jpg` corresponds to `xml/Japan/0001.xml`. If the actual RDD2022 layout differs, choose roots accordingly or prepare a matching annotation directory without modifying originals. Missing XML under an explicitly supplied annotation root and malformed XML are listed in the error CSV; an empty `damage_types` field does not establish that an image is undamaged. Object names are preserved verbatim, including D00, D10, D20 and D40.

## Sample first

Each index command processes only the first 100 paths in sorted order by default; use `--limit 25` for a smaller sample. `--dry-run` reads and checks data but writes nothing. The sample is for verifying paths, XML, provenance, and labels; it is not necessarily representative.

```sh
python -m roadsense.pipeline index --root /data/rdd/images --annotations /data/rdd/xml --source rdd2022 --manifest rdd_manifest.csv --output work/rdd.csv --errors work/rdd_errors.csv --limit 25 --dry-run
python -m roadsense.pipeline index --root /data/rdd/images --annotations /data/rdd/xml --source rdd2022 --manifest rdd_manifest.csv --output work/rdd.csv --errors work/rdd_errors.csv --limit 25
python -m roadsense.pipeline index --root /data/mapillary --source mapillary --manifest mapillary_manifest.csv --output work/mapillary.csv --errors work/mapillary_errors.csv --limit 25
python -m roadsense.pipeline index --root /data/self --source self --manifest self_manifest.csv --output work/self.csv --errors work/self_errors.csv --limit 25
python -m roadsense.pipeline combine work/rdd.csv work/mapillary.csv work/self.csv --output work/metadata.csv
python -m roadsense.pipeline audit --input work/metadata.csv --output work/audit.json
python -m roadsense.label --input work/metadata.csv
python -m roadsense.pipeline split --input work/metadata.csv --output work/splits.csv --seed 42 --dry-run
python -m roadsense.pipeline split --input work/metadata.csv --output work/splits.csv --seed 42
```

The UI saves each decision atomically to the CSV and resumes at the first unlabeled image. Back up the CSV before re-indexing: indexing writes a new CSV and does not merge existing decisions. Use Previous/Next to review decisions; Uncertain is a human decision but is excluded from usable labeled data. Use one UI process at a time. After checking the sample, repeat indexing with `--all`, combine, audit and label the full dataset. Re-indexing and combining overwrite their output files unless `--dry-run` is specified.

## Audit and splits

Audit reports SHA-256 exact duplicates and difference-hash near-duplicate candidates (default Hamming distance ≤4; change with `--threshold`). Inspect candidates: perceptual hashes can have false positives and false negatives, especially for uniform scenes. Pairwise comparison is quadratic, so large datasets may require substantial time and memory. Both exact and near candidates are joined into components during splitting, including unlabeled images, to keep them together. Use the same threshold for audit and split. Missing or inaccurate group IDs and undetected near-duplicates can still cause leakage; fill and review group metadata before trusting results.

The split reserves approximately 20% of explicitly Cambodian candidate groups (`--holdout-fraction` changes this) as a Cambodian-only holdout, then greedily assigns remaining groups toward 70/15/15 by usable image count with a seeded deterministic shuffle. These targets are approximate: large or duplicate-linked groups, small samples, and imbalanced classes can substantially change proportions. It is not stratified. A holdout requires at least two independent Cambodian components with usable labels; mixed-country duplicate-linked components are excluded from holdout eligibility. Review actual source and class distributions in each split before evaluation. The output includes all records with a `split` and `eligible` flag: use only `eligible=True` (Good/Moderate/Poor) for future supervised work; unlabeled and Uncertain records remain assigned for leakage control. Do not interpret the holdout as statistically representative without checking its locations, sessions and class balance.

Audit output includes source and label counts, and index error CSVs identify unreadable images and annotation problems. The split command prints split counts and number of rows without group IDs. Run `python -m roadsense.pipeline --help` and each subcommand's `--help` for options. Results depend on the input order, contents, grouping, seed, and threshold; retain the input CSV and settings alongside outputs.
