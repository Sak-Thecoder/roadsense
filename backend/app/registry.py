"""Load configured Keras artifacts on demand and cache a bounded number."""
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from .preprocess import CLASSES, INPUT_SIZE, MODELS, PREPROCESSING, preprocess
from .schemas import Benchmark, ModelInfo, Prediction

NAMES = dict(zip(MODELS, ("MobileNetV2", "ResNet50", "EfficientNetB0")))
log = logging.getLogger(__name__)


class Unavailable(RuntimeError):
    pass


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Registry:
    def __init__(self, model_dir: Path, result_dir: Path, enabled=MODELS, cache_size=1):
        self.model_dir, self.result_dir = model_dir, result_dir
        self.enabled = set(enabled)
        self.cache_size = max(1, cache_size)
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.failed = {}
        self.fingerprints = {}

    def metadata(self, name):
        if name not in self.enabled:
            return None
        model = self.model_dir / f"{name}.keras"
        try:
            metadata = json.loads(model.with_suffix(".json").read_text())
            if (
                not model.is_file()
                or metadata["model"] != name
                or metadata["classes"] != list(CLASSES)
                or metadata["input_size"] != list(INPUT_SIZE)
                or metadata["preprocessing"] != PREPROCESSING
                or not isinstance(metadata["version"], str)
                or metadata["sha256"] != self.digest(model)
            ):
                return None
            return metadata
        except (OSError, KeyError, ValueError, TypeError):
            return None

    def digest(self, path):
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
        metadata = self.metadata(name)
        usable = bool(metadata and self.failed.get(name) != metadata["sha256"])
        try:
            size = (self.model_dir / f"{name}.keras").stat().st_size / 1_000_000 if usable else None
        except OSError:
            size, usable = None, False
        return ModelInfo(model=name, name=NAMES[name], available=usable, size_mb=size,
                         version=metadata["version"] if metadata else None,
                         use_scope=metadata.get("use_scope") if metadata else None)

    def predict(self, name, pixels):
        metadata = self.metadata(name)
        if not metadata or self.failed.get(name) == metadata["sha256"]:
            raise Unavailable(f"{NAMES[name]} is unavailable. A compatible trained artifact is required.")
        with self.lock:
            entry = self.cache.pop(name, None)
            if not entry or entry[0] != metadata["sha256"]:
                if len(self.cache) >= self.cache_size:
                    self.cache.popitem(last=False)
                try:
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
            scores = np.asarray(model(preprocess(pixels, name)[None], training=False))
            elapsed = (time.perf_counter() - start) * 1000
        if (scores.shape != (1, 3) or not np.isfinite(scores).all()
                or (scores < 0).any() or (scores > 1).any()
                or not np.isclose(scores.sum(), 1, atol=1e-4)):
            raise RuntimeError("Invalid output tensor")
        values = scores[0].tolist()
        index = int(np.argmax(values))
        return Prediction(model=name, prediction=CLASSES[index], confidence=values[index],
                          scores=dict(zip(CLASSES, values)), inference_ms=elapsed)

    def benchmark(self, name):
        metadata = self.metadata(name)
        if not metadata:
            return None
        try:
            result = Benchmark.model_validate_json((self.result_dir / f"{name}.json").read_text())
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
