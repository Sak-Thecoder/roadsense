"""Read eligible manifests and build a shared TensorFlow input pipeline.

The "manifest" is a CSV file with one row per image: its path, label, checksum,
licence/permission status, which scene it belongs to, and which split
(train / validation / test) it is assigned to. This module:
  * checks that every row is complete and trustworthy (`candidates`);
  * checks that no scene or duplicate photo leaks between splits (`validate_partitions`);
  * turns the rows into a batched TensorFlow dataset for training (`dataset`).
"""
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
    """Read the manifest and return only the eligible rows, after checking each one is complete."""
    # "utf-8-sig" also handles CSV files saved by Excel, which start with an invisible BOM character.
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
        # A row is usable only if: the label is valid, the image could be read, we have
        # permission to use it, it has a scene group, a 64-character SHA-256 hash, and the file exists.
        if (row.get("verified_quality_label") not in CLASSES or row.get("read_status") != "readable"
                or row.get("permission_status", "").lower() not in {"confirmed", "granted", "approved"}
                or not row.get("capture_group", "").strip()
                or not re.fullmatch("[a-fA-F0-9]{64}", row.get("sha256", ""))
                or not Path(row.get("image_path", "")).is_file()):
            raise ValueError(f"Incomplete eligible record {identifier}: check label, usage permission, scene group, hash and path.")
    # Sorting makes the order (and therefore training) reproducible.
    return sorted(rows, key=lambda row: row["image_id"])


def group_keys(row):
    """Keys that must all stay inside one split: the scene, the exact file, and any near-duplicate group.

    Example: several photos of the same street taken seconds apart share a capture_group.
    If one were in train and another in test, the test score would be unrealistically high.
    """
    return [("scene", row["capture_group"]), ("exact", row["sha256"].lower())] + (
        [("similarity", row["duplicate_group"])] if row.get("duplicate_group") else [])


def partition_rows(path, pilot=False):
    """Read a manifest and return {"train": [...], "validation": [...], "test": [...], ...}."""
    return validate_partitions(candidates(path), pilot=pilot)


def validate_partitions(rows, pilot=False):
    """Group rows by split and enforce the rules that keep the test set honest."""
    result = defaultdict(list)
    owners = {}  # group key -> the split that first used it
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
        # Images the original dataset authors reserved for testing must never be trained on.
        if row.get("original_split") in {"test", "non_train"} and split not in {"test", "cambodia_holdout"}:
            raise ValueError(f"Source test image used outside a test partition: {row['image_id']}")
        if split == "cambodia_holdout" and row.get("country", "").strip().lower() not in {"kh", "cambodia"}:
            raise ValueError(f"Cambodian holdout location is unverified: {row['image_id']}")
        for key in group_keys(row):
            # setdefault records the split the first time a key is seen and returns it afterwards,
            # so a key that appears in two different splits is detected here.
            if owners.setdefault(key, split) != split:
                raise ValueError(f"Scene or duplicate group crosses partitions: {key}")
        result[split].append(row)
    # Every split must contain at least one Good, one Moderate and one Poor image.
    for split in (("train", "test") if pilot else PARTITIONS[:3]):
        if {r["verified_quality_label"] for r in result[split]} != set(CLASSES):
            raise ValueError(f"Partition {split} needs examples of each class.")
    return result


def load_pixels(row, focal=False):
    """Read one image (from disk or from inside a ZIP), verify its checksum, and decode it to 224x224 pixels."""
    path = Path(row["image_path"])
    if row.get("archive_member"):
        # The image lives inside a ZIP archive; read it directly without extracting.
        with ZipFile(path) as archive:
            content = archive.read(row["archive_member"])
    else:
        content = path.read_bytes()
    # Make sure the file is exactly the one that was reviewed and labelled.
    if hashlib.sha256(content).hexdigest() != row["sha256"].lower():
        raise ValueError(f"Image bytes differ from the manifest: {row['image_id']}")
    return decode(content, focal=focal)


def augmentation_layers(kind, seed):
    """Random image changes applied only to training images, so the model sees more variety.

    "basic": horizontal flip + slight rotation.
    "strong": also zoom, shift, brightness and contrast changes.
    """
    import tensorflow as tf

    layers = [tf.keras.layers.RandomFlip("horizontal", seed=seed),
              tf.keras.layers.RandomRotation(0.03, seed=seed + 1)]  # up to ±3% of a full turn (~11°)
    if kind == "strong":
        # Photometric jitter for lighting/camera differences, and mild zoom/shift for framing.
        layers += [tf.keras.layers.RandomZoom((-0.15, 0.0), seed=seed + 2),   # zoom in by up to 15%
                   tf.keras.layers.RandomTranslation(0.05, 0.05, seed=seed + 3),  # shift up to 5%
                   tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255), seed=seed + 4),
                   tf.keras.layers.RandomContrast(0.2, seed=seed + 5)]
    elif kind != "basic":
        raise ValueError(f"Unknown augmentation {kind!r}")
    return tf.keras.Sequential(layers)


def grade_index(row):
    """Default label: 0 = Good, 1 = Moderate, 2 = Poor."""
    return CLASSES.index(row["verified_quality_label"])


def dataset(rows, model, batch_size, training=False, seed=42, cache_decoded=False, focal=False,
            augmentation="basic", label_index=grade_index):
    """Build a tf.data pipeline: load -> (cache) -> (shuffle) -> batch -> (augment) -> normalise.

    `label_index` decides which integer label each row gets; train.py passes its own
    function when it trains on fine surface:level classes instead of the three grades.
    """
    import tensorflow as tf

    # A Python generator that yields one (image, label) pair at a time.
    # uint8 (0..255 integers) uses 4x less memory than float32 when cached.
    def examples():
        for row in rows:
            yield load_pixels(row, focal=focal).astype(np.uint8), np.int32(label_index(row))

    data = tf.data.Dataset.from_generator(examples, output_signature=(
        tf.TensorSpec((*INPUT_SIZE, 3), tf.uint8), tf.TensorSpec((), tf.int32)))
    # Tell TensorFlow how many items there are (a generator doesn't know its own length).
    data = data.apply(tf.data.experimental.assert_cardinality(len(rows)))
    if cache_decoded:
        # Keep decoded images in memory after the first epoch, so JPEGs aren't decoded again.
        data = data.cache()
    if training:
        data = data.shuffle(min(256, len(rows)), seed=seed)
    data = data.batch(batch_size)
    if training:
        augment = augmentation_layers(augmentation, seed)
        # Augment in float, then clip back to the valid 0..255 pixel range.
        data = data.map(lambda pixels, labels: (
            tf.clip_by_value(augment(tf.cast(pixels, tf.float32), training=True), 0, 255), labels))

    def transform(pixels, labels):
        # numpy_function runs our NumPy `preprocess` (shared with the API) inside the TF pipeline.
        converted = tf.numpy_function(lambda values: preprocess(values, model), [pixels], tf.float32)
        # numpy_function loses shape information; ensure_shape restores it: (batch, 224, 224, 3).
        return tf.ensure_shape(converted, (None, *INPUT_SIZE, 3)), labels

    # prefetch(1) prepares the next batch while the current one is being trained on.
    return data.map(transform).prefetch(1)
