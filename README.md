# RoadSense

RoadSense grades the road surface in a photo as **Good**, **Moderate** or **Poor**. You can take a photo or upload one in the web app, choose a model, and get the grade with the confidence for each class. Photos that don't show a road are rejected before grading.

The app runs on a React front end and a FastAPI backend, using three transfer-learning CNNs: MobileNetV2, ResNet50 and EfficientNetB0.

## Results

These are the results on the 776-image StreetSurfaceVis test set, which was held out and evaluated once.

| Model | Accuracy | Macro F1 | Size | CPU time / image |
|---|---:|---:|---:|---:|
| MobileNetV2 | 82.7% | 0.696 | 10 MB | 53 ms |
| **ResNet50** | **84.1%** | **0.732** | 96 MB | 124 ms |
| EfficientNetB0 | 80.8% | 0.702 | 18 MB | 114 ms |

- ResNet50 is the most accurate model.
- MobileNetV2 is the best choice for speed and size.
- Moderate is the hardest class for every model (F1 ≈ 0.45), because it is easily confused with Good.
- The test images come from Germany. Accuracy on Cambodian roads has not been measured properly yet.

## Run locally

You need Python 3.11 and Node.js 22.12+.

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r backend/requirements.txt
python -m backend.app.road_photo          # downloads the road-check model
uvicorn backend.app.main:app --reload     # API on http://localhost:8000
```

In a second terminal:

```sh
npm --prefix frontend ci
npm --prefix frontend run dev             # app on http://localhost:5173
```

The trained models are not included in the repo. Put them in `backend/models/` and their results in `backend/results/`, or train your own (see below).

## How it works

1. **Road check:** SegFormer-B0 makes sure at least 12% of the lower-centre of the photo is road, path or sidewalk.
2. **Preprocessing:** the photo is rotated using its EXIF data, cropped to the lower-centre road area and resized to 224×224.
3. **Classification:** each model predicts *surface type × quality level*. A fixed layer then adds these up into Good, Moderate and Poor.

## Training

```sh
pip install -r training/requirements.txt
python -m training.prepare_manifest --manifest approved.csv --output work/splits.csv
python -m training.train    --manifest work/splits.csv --model resnet50 \
  --config training/configs/quality_v3.json --output-dir work/run/models
python -m training.evaluate --manifest work/splits.csv --model resnet50 \
  --model-dir work/run/models --output-dir work/run/results
```

**Data** (full credits in [Datasets and credits](#datasets-and-credits)):
- StreetSurfaceVis: 9,122 German street images, mapped from 5 smoothness levels to 3 classes.
- 50 self-collected Cambodian photos, used with the owner's permission.

The split groups photos by location, capture time and near-duplicates, so no scene appears in both training and test.

**Training setup:**
1. Train a new classification head with the backbone frozen.
2. Fine-tune the whole network with BatchNorm frozen.
3. Throughout, use class weights and label smoothing 0.1, and pick the best epoch by validation macro F1.

Images and trained models are not included in this repo.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Service status |
| GET | `/models` | Available models |
| POST | `/predict/{model}` | Grade one photo (multipart field `image`) |
| POST | `/compare` | Grade with all models |
| GET | `/benchmark` | Test-set results |

Model IDs are `mobilenetv2`, `resnet50` and `efficientnetb0`. Interactive docs are at `http://localhost:8000/docs`.

## Tests

```sh
pip install -r backend/requirements-dev.txt
python -m pytest backend/tests -q
npm --prefix frontend run test
npm --prefix frontend run build
```

## Project layout

```
backend/        FastAPI app, model registry, road check
frontend/       React + Vite web app
training/       dataset splitting, training and evaluation
investigation/  error analysis, Grad-CAM, calibration
roadsense/      dataset audit and labelling tools
integration/    dataset integration scripts and reports
docs/           dataset notes, Cambodian pilot, deployment
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Docker and hosting.

## Datasets and credits

RoadSense builds on other people's work. Thank you to:

- **StreetSurfaceVis:** the main training and test data, by Alexandra Kapp, Edith Hoffmann, Esther Weigmann and Helena Mihaljević. The images are crowdsourced from [Mapillary](https://www.mapillary.com/) contributors and released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
  - Dataset: [Zenodo record 11449977](https://zenodo.org/records/11449977)
  - Paper: A. Kapp, E. Hoffmann, E. Weigmann and H. Mihaljević, "StreetSurfaceVis: a dataset of crowdsourced street-level imagery annotated by road surface type and quality," *Scientific Data*, 2025 ([open access](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739464/))
- **Cambodian road photos:** taken and labelled by the RoadSense team.
- **SegFormer-B0 (road check):** by E. Xie et al. (NVIDIA), trained on the [ADE20K](https://ade20k.csail.mit.edu/) scene-parsing dataset. We use the ONNX export [Xenova/segformer-b0-finetuned-ade-512-512](https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512).
- **ImageNet weights:** MobileNetV2, ResNet50 and EfficientNetB0 start from ImageNet-pretrained weights in [Keras Applications](https://keras.io/api/applications/).

The licence below covers the code only. The images and pretrained models keep their own licences, so check them before reusing or sharing anything built from them.

## License

The code is released under the [MIT License](LICENSE).
