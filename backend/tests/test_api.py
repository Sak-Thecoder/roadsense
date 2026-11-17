import json
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import create_app
from backend.app.preprocess import CLASSES, INPUT_SIZE, PREPROCESSING, decode, preprocess
from backend.app.registry import Registry, file_hash
from backend.tests.fixture_server import fixtures


def jpeg():
    stream = BytesIO()
    Image.new("RGB", (32, 24), (240, 15, 10)).save(stream, format="JPEG")
    return stream.getvalue()


def upload(bytes_):
    return {"image": ("road.jpg", bytes_, "image/jpeg")}


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    return fixtures(tmp_path_factory.mktemp("keras"), benchmarks=True)


def test_missing_models_and_benchmark(tmp_path):
    client = TestClient(create_app(Registry(tmp_path, tmp_path)))
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
    client = TestClient(create_app(registry))
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
    client = TestClient(create_app(registry))
    assert client.post("/predict/mobilenetv2", files=upload(jpeg())).status_code == 503
    assert not registry.info("mobilenetv2").available
