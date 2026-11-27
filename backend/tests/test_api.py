import json
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.app.preprocess import CLASSES, FOCAL_PREPROCESSING, INPUT_SIZE, PREPROCESSING, decode, preprocess
from backend.app.registry import Registry, file_hash
from backend.app.road_photo import RoadPhotoValidator, road_coverage
from backend.tests.fixture_server import check_fixture_photo, fixtures


def jpeg(color=(240, 15, 10)):
    stream = BytesIO()
    Image.new("RGB", (32, 24), color).save(stream, format="JPEG")
    return stream.getvalue()


def upload(bytes_):
    return {"image": ("road.jpg", bytes_, "image/jpeg")}


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    return fixtures(tmp_path_factory.mktemp("keras"), benchmarks=True)


def test_missing_models_and_benchmark(tmp_path):
    client = TestClient(create_app(Registry(tmp_path, tmp_path), photo_validator=check_fixture_photo))
    assert client.get("/health").json()["available_models"] == []
    assert client.get("/benchmark").json() == []
    assert client.get("/benchmark/mobilenetv2").status_code == 404
    assert client.post("/predict/mobilenetv2", files=upload(jpeg())).status_code == 503
    assert client.post("/predict/unknown", files=upload(jpeg())).status_code == 404


@pytest.mark.parametrize("path", ["/predict/mobilenetv2", "/compare"])
def test_upload_errors(tmp_path, path):
    client = TestClient(create_app(Registry(tmp_path, tmp_path), max_upload=1000))
    assert client.post(path).status_code == 422
    assert client.post(path, files=upload(b"bad jpg")).status_code == 422
    assert client.post(path, files=upload(b"x" * 1001)).status_code == 413
    assert client.post(path, files=upload(b"x" * 90_000)).status_code == 413


def test_real_fixture_inference_and_benchmark(registry):
    client = TestClient(create_app(registry, photo_validator=check_fixture_photo))
    assert len(client.get("/health").json()["available_models"]) == 3
    response = client.post("/predict/mobilenetv2", files=upload(jpeg()))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["prediction"] == "Good"
    assert list(result["scores"]) == list(CLASSES)
    assert sum(result["scores"].values()) == pytest.approx(1.0)
    instance = registry.cache["mobilenetv2"][1]
    compare = client.post("/compare", files=upload(jpeg())).json()
    assert not compare["errors"]
    assert len(compare["predictions"]) == 3
    assert registry.cache["mobilenetv2"][1] is instance
    data = client.get("/benchmark/mobilenetv2").json()
    assert data["dataset_size"] == 3 and sum(map(sum, data["confusion_matrix"])) == 3
    assert data["model_version"] == "isolated-test-fixture"


@pytest.mark.parametrize("path", ["/predict/mobilenetv2", "/compare"])
def test_non_road_and_unavailable_check_never_reach_classifiers(tmp_path, path):
    class NoPredictions:
        def predict(self, *args):
            pytest.fail("The quality classifier received a photo without passing the road check.")

    rejected = TestClient(create_app(NoPredictions(), photo_validator=check_fixture_photo))
    response = rejected.post(path, files=upload(jpeg((200, 200, 200))))
    assert response.status_code == 422
    assert response.json()["detail"].startswith("Invalid road photo.")

    unavailable = TestClient(create_app(NoPredictions(), photo_validator=RoadPhotoValidator(tmp_path / "missing.onnx")))
    assert unavailable.post(path, files=upload(jpeg())).status_code == 503


def test_road_coverage_uses_confident_surfaces_in_lower_center():
    logits = np.zeros((150, 8, 8), dtype=np.float32)
    logits[6, :4, :] = 9  # A road outside the area classified by RoadSense.
    logits[12, 4:, :] = 9  # Person in the foreground.
    assert road_coverage(logits) == 0
    logits[6, 4:, 2:4] = 12
    assert road_coverage(logits) == 0.5


def test_road_coverage_counts_earth_for_unpaved_roads():
    logits = np.zeros((150, 8, 8), dtype=np.float32)
    logits[13, 4:, 2:6] = 12  # "earth, ground": how the model often labels gravel and dirt roads.
    assert road_coverage(logits) == 1
    logits[13] = 0
    logits[9, 4:, 2:6] = 12  # Grass alone is not a road surface.
    assert road_coverage(logits) == 0


def test_preprocess_matches_keras():
    from tensorflow.keras.applications import efficientnet, mobilenet_v2, resnet50

    pixels = decode(jpeg())
    assert pixels.shape == (224, 224, 3)
    for name, application in (("mobilenetv2", mobilenet_v2), ("resnet50", resnet50),
                              ("efficientnetb0", efficientnet)):
        np.testing.assert_allclose(preprocess(pixels, name), application.preprocess_input(pixels.copy()))


def test_exif_orientation():
    stream = BytesIO()
    pixels = Image.new("RGB", (10, 20))
    pixels.paste((255, 0, 0), (0, 0, 10, 10))
    exif = pixels.getexif()
    exif[274] = 6
    pixels.save(stream, "JPEG", exif=exif)
    decoded = decode(stream.getvalue())
    assert decoded[112, -5, 0] > 200
    assert decoded[112, 5, 0] < 30


def test_corrupt_model_is_rejected(tmp_path):
    model = tmp_path / "mobilenetv2.keras"
    model.write_bytes(b"not a model")
    model.with_suffix(".json").write_text(json.dumps(dict(
        model="mobilenetv2", sha256=file_hash(model), classes=list(CLASSES),
        input_size=list(INPUT_SIZE), preprocessing=PREPROCESSING, version="invalid-fixture"
    )))
    registry = Registry(tmp_path, tmp_path)
    assert registry.info("mobilenetv2").available
    client = TestClient(create_app(registry, photo_validator=check_fixture_photo))
    assert client.post("/predict/mobilenetv2", files=upload(jpeg())).status_code == 503
    assert not registry.info("mobilenetv2").available


def test_focal_crop_uses_full_resolution_lower_centre():
    image = Image.new("RGB", (400, 800), (0, 0, 0))
    image.paste((255, 255, 255), (100, 400, 300, 800))  # exactly the focal region
    image.paste((0, 255, 0), (100, 400, 102, 800))  # 2px line: 0.5% of full width, 1% of the crop
    stream = BytesIO()
    image.save(stream, "PNG")
    focal = decode(stream.getvalue(), focal=True)
    full = decode(stream.getvalue())
    assert focal.shape == (*INPUT_SIZE, 3)
    assert focal[:, 10:].min() > 250  # only the road region remains
    assert full[:100].max() < 5  # full frame still contains the upper background
    assert focal[:, 1, 1].mean() > focal[:, 1, 0].mean() + 100  # the thin line survives the crop


def test_registry_applies_focal_crop_per_model(tmp_path):
    import tensorflow as tf

    inputs = tf.keras.Input((*INPUT_SIZE, 3))
    head = tf.keras.layers.Dense(3, activation="softmax")
    model = tf.keras.Model(inputs, head(tf.keras.layers.GlobalAveragePooling2D()(inputs)))
    head.set_weights([np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)])
    path = tmp_path / "mobilenetv2.keras"
    model.save(path)
    path.with_suffix(".json").write_text(json.dumps(dict(
        model="mobilenetv2", version="focal", sha256=file_hash(path), classes=list(CLASSES),
        input_size=list(INPUT_SIZE), preprocessing=FOCAL_PREPROCESSING)))
    registry = Registry(tmp_path, tmp_path)
    assert registry.info("mobilenetv2").available
    image = Image.new("RGB", (400, 800), (0, 0, 255))
    image.paste((255, 0, 0), (100, 400, 300, 800))
    # A full-frame resize would be mostly blue (Poor); the focal crop is all red (Good).
    assert registry.predict("mobilenetv2", image).prediction == "Good"
    with pytest.raises(ValueError, match="full-resolution"):
        registry.predict("mobilenetv2", np.zeros((*INPUT_SIZE, 3), np.float32))
