"""Load configured Keras artifacts on demand and cache a bounded number.

The Registry is the API's "model manager". For every model it:
  1. checks that the .keras file and its .json description match (right classes,
     input size, preprocessing and SHA-256 checksum), so a wrong or half-copied
     file is never used;
  2. loads the model into memory only when it is first needed and keeps a small
     cache, because ResNet50 alone is about 100 MB;
  3. runs predictions and checks that the output looks like valid probabilities.
"""
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import ValidationError

from .preprocess import CLASSES, INPUT_SIZE, MODELS, PREPROCESSING_MODES, image_pixels, preprocess, uses_focal_crop
from .schemas import Benchmark, ModelInfo, Prediction

# Maps internal IDs to display names, e.g. {"mobilenetv2": "MobileNetV2", ...}.
NAMES = dict(zip(MODELS, ("MobileNetV2", "ResNet50", "EfficientNetB0")))
log = logging.getLogger(__name__)


class Unavailable(RuntimeError):
    """Raised when a model file is missing, invalid, or failed to load."""


def file_hash(path: Path) -> str:
    """SHA-256 fingerprint of a file. Any change to the file gives a different hash."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Registry:
    def __init__(self, model_dir: Path, result_dir: Path, enabled=MODELS, cache_size=1):
        self.model_dir, self.result_dir = model_dir, result_dir
        self.enabled = set(enabled)
        self.cache_size = max(1, cache_size)
        # OrderedDict remembers insertion order, so the oldest model can be evicted first
        # (a simple "least recently used" cache).
        self.cache = OrderedDict()
        # RLock = a lock the same thread may take twice. It stops two web requests
        # from loading or evicting models at the same time.
        self.lock = threading.RLock()
        # name -> sha256 of a model file that failed to load (so we don't retry it on every request).
        self.failed = {}
        # path -> ((modified_time, size), sha256): avoids re-hashing a 100 MB file on every request.
        self.fingerprints = {}

    def metadata(self, name):
        """Return the model's .json metadata if the model is enabled and valid, otherwise None."""
        if name not in self.enabled:
            return None
        model = self.model_dir / f"{name}.keras"
        try:
            metadata = json.loads(model.with_suffix(".json").read_text())
            # Every check must pass, otherwise the model is treated as unavailable.
            # The sha256 check ensures the .json really describes this exact .keras file.
            if (
                not model.is_file()
                or metadata["model"] != name
                or metadata["classes"] != list(CLASSES)
                or metadata["input_size"] != list(INPUT_SIZE)
                or metadata["preprocessing"] not in PREPROCESSING_MODES
                or not isinstance(metadata["version"], str)
                or metadata["sha256"] != self.digest(model)
            ):
                return None
            return metadata
        except (OSError, KeyError, ValueError, TypeError):
            return None

    def digest(self, path):
        """SHA-256 of a file, recomputed only when its modification time or size changes."""
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        with self.lock:
            cached = self.fingerprints.get(path)
            if cached and cached[0] == key:
                return cached[1]
            digest = file_hash(path)
            self.fingerprints[path] = (key, digest)
            return digest

    def info(self, name):
        """Summary used by GET /models: is the model usable, how big is it, which version."""
        metadata = self.metadata(name)
        # Usable = valid metadata AND this exact file has not previously failed to load.
        usable = bool(metadata and self.failed.get(name) != metadata["sha256"])
        try:
            size = (self.model_dir / f"{name}.keras").stat().st_size / 1_000_000 if usable else None
        except OSError:
            size, usable = None, False
        return ModelInfo(model=name, name=NAMES[name], available=usable, size_mb=size,
                         version=metadata["version"] if metadata else None,
                         use_scope=metadata.get("use_scope") if metadata else None)

    def predict(self, name, image):
        """Classify a decoded PIL image (or, for full-frame models, pre-resized pixels)."""
        metadata = self.metadata(name)
        if not metadata or self.failed.get(name) == metadata["sha256"]:
            raise Unavailable(f"{NAMES[name]} is unavailable. A compatible trained artifact is required.")
        # Prepare the 224x224 input the same way this model was trained.
        focal = uses_focal_crop(metadata["preprocessing"])
        if isinstance(image, Image.Image):
            pixels = image_pixels(image, focal=focal)
        elif focal:
            raise ValueError("Focal-crop models need the decoded full-resolution image.")
        else:
            pixels = image
        with self.lock:
            # Take the model out of the cache (it is put back at the end = "most recently used").
            entry = self.cache.pop(name, None)
            # Reload if it is not cached yet or the file on disk has changed (different sha256).
            if not entry or entry[0] != metadata["sha256"]:
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)  # evict the oldest cached model to free memory
                try:
                    # Imported here (not at the top) so the API starts quickly; TensorFlow is slow to import.
                    import tensorflow as tf
                    model = tf.keras.models.load_model(self.model_dir / f"{name}.keras", compile=False)
                except Exception as exc:
                    self.failed[name] = metadata["sha256"]
                    log.exception("Unable to load model %s", name)
                    raise Unavailable(f"{NAMES[name]} could not be loaded.") from exc
                entry = (metadata["sha256"], model)
            self.cache[name] = entry
            model = entry[1]
            start = time.perf_counter()
            # [None] adds a batch dimension: (224, 224, 3) -> (1, 224, 224, 3), because Keras
            # models always take a batch of images. training=False disables dropout/augmentation.
            scores = np.asarray(model(preprocess(pixels, name)[None], training=False))
            elapsed = (time.perf_counter() - start) * 1000
        # Sanity check: output must be one row of 3 probabilities between 0 and 1 that sum to 1.
        if (scores.shape != (1, 3) or not np.isfinite(scores).all()
                or (scores < 0).any() or (scores > 1).any()
                or not np.isclose(scores.sum(), 1, atol=1e-4)):
            raise RuntimeError("Invalid output tensor")
        values = scores[0].tolist()
        # The predicted grade is the class with the highest probability.
        index = int(np.argmax(values))
        return Prediction(model=name, prediction=CLASSES[index], confidence=values[index],
                          scores=dict(zip(CLASSES, values)), inference_ms=elapsed)

    def benchmark(self, name):
        """Load saved test-set results, but only if they belong to the model currently installed."""
        metadata = self.metadata(name)
        if not metadata:
            return None
        try:
            result = Benchmark.model_validate_json((self.result_dir / f"{name}.json").read_text())
            # The result must come from this exact model file (same sha256 and version) and
            # be internally consistent: a 3x3 confusion matrix whose cells add up to the dataset size.
            if (result.model == name and result.model_sha256 == metadata["sha256"]
                    and result.model_version == metadata["version"] and result.classes == list(CLASSES)
                    and result.use_scope == metadata.get("use_scope")
                    and set(result.per_class) == set(CLASSES)
                    and all(len(row) == 3 and min(row) >= 0 for row in result.confusion_matrix)
                    and sum(map(sum, result.confusion_matrix)) == result.dataset_size):
                return result
        except (OSError, ValidationError):
            pass
        return None
