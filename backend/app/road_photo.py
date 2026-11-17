"""Check for a visible road surface before running a road-quality classifier."""
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.request import urlopen

import numpy as np

from .preprocess import InvalidImage, image_pixels
from .registry import file_hash

MODEL_URL = (
    "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/"
    "d3e5499fa8701ff0453ca940a8dfeae39b2f1504/onnx/model_quantized.onnx"
)
MODEL_SHA256 = "9a98d6daf3d926869ab8cc4c2ed7374a2bc23b889bb7ca3b0915d15e3c4756bb"
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Zero-based ADE20K labels for road, sidewalk, path, and dirt track.
SURFACE_LABELS = (6, 11, 52, 91)


class RoadPhotoCheckUnavailable(RuntimeError):
    """The road-photo check could not safely evaluate this image."""


def model_path():
    default = Path(__file__).resolve().parents[1] / "models" / "road-photo-segformer.onnx"
    return Path(os.getenv("ROADSENSE_ROAD_PHOTO_MODEL", default))


def road_coverage(logits):
    """Measure confident surface pixels in the classifier's lower-center view."""
    if logits.ndim != 3 or logits.shape[0] != 150 or min(logits.shape[1:]) < 4:
        raise ValueError("Unexpected road-photo model output shape.")
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite road-photo model output.")
    _, height, width = logits.shape
    focal = logits[:, height // 2:, width // 4:3 * width // 4]
    winners = focal.argmax(axis=0)
    # Softmax probability of the winning class at each pixel.
    confidence = 1 / np.exp(focal - focal.max(axis=0)).sum(axis=0)
    return float(np.mean(np.isin(winners, SURFACE_LABELS) & (confidence >= 0.45)))


class RoadPhotoValidator:
    def __init__(self, path):
        self.path = path
        self._session = None
        self._lock = Lock()

    def _load(self):
        with self._lock:
            if self._session is None:
                if file_hash(self.path) != MODEL_SHA256:
                    raise ValueError("Road-photo model checksum does not match.")
                import onnxruntime as ort

                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                self._session = ort.InferenceSession(str(self.path), options, providers=["CPUExecutionProvider"])
            return self._session

    def __call__(self, image):
        try:
            pixels = (image_pixels(image, (512, 512)) / 255 - IMAGE_MEAN) / IMAGE_STD
            inputs = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
            logits = self._load().run(["logits"], {"pixel_values": inputs})[0][0]
            coverage = road_coverage(logits)
        except Exception as exc:
            raise RoadPhotoCheckUnavailable("The road-photo check is temporarily unavailable. Try again later.") from exc
        if coverage < 0.12:
            raise InvalidImage("Invalid road photo. Choose an image with a clear road surface in the lower center.")


def install_model(path):
    if path.is_file() and file_hash(path) == MODEL_SHA256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".onnx", delete=False) as temporary:
        staged = Path(temporary.name)
        try:
            with urlopen(MODEL_URL, timeout=30) as source:
                copyfileobj(source, temporary)
            temporary.close()
            if file_hash(staged) != MODEL_SHA256:
                raise ValueError("Downloaded road-photo model checksum does not match.")
            staged.chmod(0o644)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


if __name__ == "__main__":
    path = model_path()
    install_model(path)
    print(f"Road-photo model ready at {path}")
