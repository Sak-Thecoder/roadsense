from collections import Counter

import numpy as np
import pytest

from backend.app.preprocess import CLASSES, MODELS
from training.prepare_manifest import make_splits
from training.preprocess import candidates, validate_partitions
from training.train import build_model


def test_empty_manifest_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("image_id,eligible\n")
    with pytest.raises(ValueError, match="No eligible"):
        candidates(path)


def test_grouped_splits_and_source_test_boundary():
    rows = []
    for index in range(18):
        for label in CLASSES:
            rows.append(dict(image_id=f"{index}-{label}", capture_group=str(index), sha256=f"{index}-{label}",
                             verified_quality_label=label, original_split="test" if index < 3 else "train"))
    result = make_splits(rows, 42, 10)
    assert result == make_splits(rows, 42, 10)
    assert Counter(r["split"] for r in result)["test"] == 9
    assert all(r["original_split"] == "test" for r in result if r["split"] == "test")
    assert all(len({r["split"] for r in result if r["capture_group"] == str(i)}) == 1 for i in range(18))
    rows[9]["sha256"] = rows[0]["sha256"]
    with pytest.raises(ValueError, match="source test"):
        make_splits(rows, 42, 10)


def test_train_only_cam_collection_keeps_author_test_independent():
    rows = [
        dict(image_id=f"{index}-{label}", capture_group=str(index), sha256=f"{index}-{label}",
             verified_quality_label=label, original_split="test" if index == 0 else "train")
        for index in range(20) for label in CLASSES
    ]
    local = dict(image_id="CAM:local", capture_group="cam", sha256="cam", verified_quality_label="Good",
                 split_policy="train_only", country="Cambodia (unverified)")
    partitions = validate_partitions(make_splits(rows + [local], seed=42, cambodia_min_groups=10))
    assert partitions["test"] == [dict(r, split="test") for r in rows[:3]]
    assert dict(local, split="train") in partitions["train"]
    with pytest.raises(ValueError, match="Train-only image"):
        validate_partitions([dict(local, split="validation")])


@pytest.mark.parametrize("name", MODELS)
def test_transfer_head_forward_and_optimizer_step(name):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    model, backbone = build_model(name, dropout=0.25, weights=None, head_units=8)
    assert not backbone.trainable
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    pixels = np.zeros((1, 224, 224, 3), dtype=np.float32)
    assert np.isfinite(model.train_on_batch(pixels, np.array([1])))
    assert model(pixels, training=False).shape == (1, 3)


@pytest.mark.parametrize("selection_metric,feature_passes,improved", [
    ("val_accuracy", 2, False), ("val_loss", 0, False), ("val_macro_f1", 2, True), ("val_macro_f1", 0, True),
    ("val_macro_f1", 2, "fine"), ("val_macro_f1", 0, "fine")])
def test_train_and_evaluate_artifacts_from_isolated_manifest(tmp_path, monkeypatch, selection_metric, feature_passes,
                                                             improved):
    import csv
    import json
    import sys

    import tensorflow as tf
    from PIL import Image

    from backend.app.registry import Registry, file_hash
    from training import evaluate, train

    rows = []
    for partition_index, split in enumerate(("train", "validation", "test")):
        for label_index, label in enumerate(CLASSES):
            path = tmp_path / f"{split}-{label}.png"
            rgb = tuple(230 - partition_index if index == label_index else 12 for index in range(3))
            Image.new("RGB", (35, 24), rgb).save(path)
            for surface in (("asphalt", "sett") if improved == "fine" else ("",)):
                rows.append(dict(image_id=f"{path.stem}-{surface}", image_path=str(path), sha256=file_hash(path),
                                 verified_quality_label=label, read_status="readable", eligible="True",
                                 permission_status="confirmed", capture_group=f"{path.stem}-{surface}",
                                 split=split, source="streetSurfaceVis", surface_type=surface,
                                 original_label=("good", "intermediate", "bad")[label_index]))
    manifest = tmp_path / "fixture.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def tiny_classifier(*_args, outputs=3, **_kwargs):
        input_shape = (224, 224, 3)
        features = tf.keras.Sequential([
            tf.keras.Input(input_shape), tf.keras.layers.Conv2D(4, 1)
        ])
        features.trainable = False
        inputs = tf.keras.Input(input_shape)
        pooled = tf.keras.layers.GlobalAveragePooling2D(name="road_features")(features(inputs, training=False))
        scores = tf.keras.layers.Dense(outputs, activation="softmax")(pooled)
        return tf.keras.Model(inputs, scores), features

    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(head_epochs=1, fine_tune_epochs=1, fine_tune_layers=1,
        dropout=0.25, head_lr=0.001, fine_tune_lr=0.00001, seed=42, patience=2, batch_size=3,
        selection_metric=selection_metric, feature_passes=feature_passes, cache_decoded=bool(feature_passes),
        **(dict(focal_crop=True, augmentation="strong", class_weight="balanced") if improved else {}),
        **(dict(label_scheme="surface_x_level", label_smoothing=0.1) if improved == "fine" else {}))))
    models, results = tmp_path / "models", tmp_path / "results"
    monkeypatch.setattr(train, "build_model", tiny_classifier)
    monkeypatch.setattr(sys, "argv", ["train", "--manifest", str(manifest), "--model", "mobilenetv2",
                                      "--config", str(config), "--output-dir", str(models)])
    train.main()
    monkeypatch.setattr(sys, "argv", ["evaluate", "--manifest", str(manifest), "--model", "mobilenetv2",
                                      "--model-dir", str(models), "--output-dir", str(results),
                                      "--warmups", "1", "--repeats", "2"])
    evaluate.main()
    registry = Registry(models, results)
    metadata = registry.metadata("mobilenetv2")
    selection = metadata["validation_selection"]
    assert selection["metric"] == selection_metric
    assert 0 <= selection["val_accuracy"] <= 1 and 0 <= selection["val_macro_f1"] <= 1
    assert "val_macro_f1" in metadata["history"]["head"]
    from backend.app.preprocess import FOCAL_PREPROCESSING, PREPROCESSING
    assert metadata["preprocessing"] == (FOCAL_PREPROCESSING if improved else PREPROCESSING)
    metrics = registry.benchmark("mobilenetv2")
    assert metrics is not None and metrics.dataset_size == (6 if improved == "fine" else 3)
    if improved == "fine":
        assert metadata["label_scheme"] == "surface_x_level" and len(metadata["training_classes"]) == 6
        assert 0 <= selection["val_grade_accuracy"] <= 1
        saved = tf.keras.models.load_model(models / "mobilenetv2.keras", compile=False)
        scores = np.asarray(saved(np.zeros((2, 224, 224, 3), np.float32)))
        assert scores.shape == (2, 3) and np.allclose(scores.sum(axis=1), 1, atol=1e-5)
    assert (results / "mobilenetv2_confusion.png").is_file()
    assert (results / "mobilenetv2_metrics.png").is_file()


def test_capture_date_pilot_requires_explicit_mode_and_keeps_test_separate(tmp_path):
    import csv

    from backend.app.registry import file_hash
    from training.preprocess import PILOT_SCOPE, partition_rows

    rows = []
    for split in ("train", "test"):
        for label in CLASSES:
            item = tmp_path / f"{split}-{label.lower()}.jpg"
            item.write_bytes(item.name.encode())
            rows.append(dict(image_id=item.stem, image_path=str(item), sha256=file_hash(item),
                             verified_quality_label=label, read_status="readable", eligible="True",
                             permission_status="confirmed", capture_group=f"date:{split}",
                             pilot_scope=PILOT_SCOPE, split=split))
    manifest = tmp_path / "pilot.csv"

    def write_rows():
        with manifest.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows()
    with pytest.raises(ValueError, match="explicit capture-date pilot mode"):
        partition_rows(manifest)
    assert {name: len(group) for name, group in partition_rows(manifest, pilot=True).items()} == {
        "train": 3, "test": 3,
    }

    rows[-1]["sha256"] = rows[0]["sha256"]
    write_rows()
    with pytest.raises(ValueError, match="duplicate group crosses partitions"):
        partition_rows(manifest, pilot=True)
