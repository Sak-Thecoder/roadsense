"""Read eligible manifests and build a shared TensorFlow input pipeline."""
import csv
import hashlib
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from backend.app.preprocess import CLASSES, INPUT_SIZE, decode, preprocess

PILOT_SCOPE = "cambodian_capture_date_pilot"
PARTITIONS = ("train", "validation", "test", "cambodia_holdout")


def candidates(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        rows = [r for r in csv.DictReader(file) if r.get("eligible", "").lower() == "true"]
    if not rows:
        raise ValueError("No eligible images in the manifest. Resolve the dataset audit blockers first.")
    seen = set()
    for row in rows:
        identifier = row["image_id"]
        if not identifier or identifier in seen:
            raise ValueError(f"Missing or duplicate eligible image ID {identifier!r}")
        seen.add(identifier)
        if (row.get("verified_quality_label") not in CLASSES or row.get("read_status") != "readable"
                or row.get("permission_status", "").lower() not in {"confirmed", "granted", "approved"}
                or not row.get("capture_group", "").strip()
                or not re.fullmatch("[a-fA-F0-9]{64}", row.get("sha256", ""))
                or not Path(row.get("image_path", "")).is_file()):
            raise ValueError(f"Incomplete eligible record {identifier}: check label, usage permission, scene group, hash and path.")
    return sorted(rows, key=lambda row: row["image_id"])


def group_keys(row):
    return [("scene", row["capture_group"]), ("exact", row["sha256"].lower())] + (
        [("similarity", row["duplicate_group"])] if row.get("duplicate_group") else [])


def partition_rows(path, pilot=False):
    return validate_partitions(candidates(path), pilot=pilot)


def validate_partitions(rows, pilot=False):
    result = defaultdict(list)
    owners = {}
    for row in rows:
        is_pilot_row = row.get("pilot_scope") == PILOT_SCOPE
        if is_pilot_row != pilot:
            raise ValueError("A Cambodian pilot manifest requires explicit capture-date pilot mode.")
        split = row.get("split")
        if pilot and split not in {"train", "test"}:
            raise ValueError("Capture-date pilot mode accepts only train and test partitions.")
        if split not in PARTITIONS:
            raise ValueError("Run training.prepare_manifest first or supply explicit valid partitions.")
        if row.get("split_policy") == "train_only" and split != "train":
            raise ValueError(f"Train-only image used outside training: {row['image_id']}")
        if row.get("original_split") in {"test", "non_train"} and split not in {"test", "cambodia_holdout"}:
            raise ValueError(f"Source test image used outside a test partition: {row['image_id']}")
        if split == "cambodia_holdout" and row.get("country", "").strip().lower() not in {"kh", "cambodia"}:
            raise ValueError(f"Cambodian holdout location is unverified: {row['image_id']}")
        for key in group_keys(row):
            if owners.setdefault(key, split) != split:
                raise ValueError(f"Scene or duplicate group crosses partitions: {key}")
        result[split].append(row)
    for split in (("train", "test") if pilot else PARTITIONS[:3]):
        if {r["verified_quality_label"] for r in result[split]} != set(CLASSES):
            raise ValueError(f"Partition {split} needs examples of each class.")
    return result


def load_pixels(row, focal=False):
    path = Path(row["image_path"])
    if row.get("archive_member"):
        with ZipFile(path) as archive:
            content = archive.read(row["archive_member"])
    else:
        content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row["sha256"].lower():
        raise ValueError(f"Image bytes differ from the manifest: {row['image_id']}")
    return decode(content, focal=focal)


def augmentation_layers(kind, seed):
    import tensorflow as tf

    layers = [tf.keras.layers.RandomFlip("horizontal", seed=seed),
              tf.keras.layers.RandomRotation(0.03, seed=seed + 1)]
    if kind == "strong":
        # Photometric jitter for lighting/camera differences, and mild zoom/shift for framing.
        layers += [tf.keras.layers.RandomZoom((-0.15, 0.0), seed=seed + 2),
                   tf.keras.layers.RandomTranslation(0.05, 0.05, seed=seed + 3),
                   tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed + 4),
                   tf.keras.layers.RandomContrast(0.2, seed=seed + 5)]
    elif kind != "basic":
        raise ValueError(f"Unknown augmentation {kind!r}")
    return tf.keras.Sequential(layers)


def dataset(rows, model, batch_size, training=False, seed=42, cache_decoded=False, focal=False,
            augmentation="basic"):
    import tensorflow as tf

    def examples():
        for row in rows:
            yield load_pixels(row, focal=focal).astype(np.uint8), np.int32(CLASSES.index(row["verified_quality_label"]))

    data = tf.data.Dataset.from_generator(examples, output_signature=(
        tf.TensorSpec((*INPUT_SIZE, 3), tf.uint8), tf.TensorSpec((), tf.int32)))
    data = data.apply(tf.data.experimental.assert_cardinality(len(rows)))
    if cache_decoded:
        data = data.cache()
    if training:
        data = data.shuffle(min(256, len(rows)), seed=seed)
    data = data.batch(batch_size)
    if training:
        augment = augmentation_layers(augmentation, seed)
        data = data.map(lambda pixels, labels: (
            tf.clip_by_value(augment(tf.cast(pixels, tf.float32), training=True), 0, 255), labels))

    def transform(pixels, labels):
        converted = tf.numpy_function(lambda values: preprocess(values, model), [pixels], tf.float32)
        return tf.ensure_shape(converted, (None, *INPUT_SIZE, 3)), labels

    return data.map(transform).prefetch(1)
