"""Isolated Keras fixture for local automated tests only."""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from backend.app.main import create_app
from backend.app.preprocess import CLASSES, INPUT_SIZE, MODELS, PREPROCESSING
from backend.app.registry import Registry, file_hash


def fixtures(directory, benchmarks=False):
    import tensorflow as tf

    models = directory / "models"
    results = directory / "results"
    models.mkdir()
    results.mkdir()
    for index, name in enumerate(MODELS):
        inputs = tf.keras.Input((*INPUT_SIZE, 3))
        avg = tf.keras.layers.GlobalAveragePooling2D()(inputs)
        head = tf.keras.layers.Dense(3, activation="softmax", name="test_head")
        model = tf.keras.Model(inputs, head(avg))
        kernel = np.eye(3, dtype=np.float32) * (1.0 if index == 0 else 0.01)
        head.set_weights([kernel, np.zeros(3, dtype=np.float32)])
        path = models / f"{name}.keras"
        model.save(path)
        metadata = dict(model=name, version="isolated-test-fixture", sha256=file_hash(path),
                        classes=list(CLASSES), input_size=list(INPUT_SIZE), preprocessing=PREPROCESSING)
        path.with_suffix(".json").write_text(json.dumps(metadata))
    if benchmarks:
        from training.evaluate import evaluate

        rows = []
        for index, label in enumerate(CLASSES):
            photo = directory / f"sample-{index}.jpg"
            pixels = tuple(230 if channel == index else 12 for channel in range(3))
            Image.new("RGB", (48, 48), pixels).save(photo, quality=100, subsampling=0)
            rows.append(dict(image_id=f"fixture-{index}", image_path=str(photo), sha256=file_hash(photo),
                             verified_quality_label=label))
        manifest = directory / "fixture.csv"
        with manifest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for name in MODELS:
            artifact = models / f"{name}.keras"
            model = tf.keras.models.load_model(artifact, compile=False)
            metadata = json.loads(artifact.with_suffix(".json").read_text())
            metrics = evaluate(model, name, rows, metadata, artifact, manifest, "test", 2, 4)
            (results / f"{name}.json").write_text(metrics.model_dump_json())
    return Registry(models, results, cache_size=3)


if __name__ == "__main__":
    import uvicorn

    with TemporaryDirectory(prefix="roadsense-e2e-fixtures-") as temporary:
        app = create_app(fixtures(Path(temporary), benchmarks=True))
        uvicorn.run(app, host="127.0.0.1", port=8129)
