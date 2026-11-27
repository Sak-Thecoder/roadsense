"""Check for a visible road surface before running a road-quality classifier.

The quality classifiers always answer Good, Moderate or Poor, even for a photo of
a cat. To avoid meaningless answers we first run a small semantic-segmentation
model (SegFormer-B0, trained on the ADE20K dataset). It labels every pixel
(road, sky, car, tree, ...). If too few pixels in the lower-centre area are road
or a similar surface, the upload is rejected.
"""
import os
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from threading import Lock
from urllib.request import urlopen

import numpy as np

from .preprocess import InvalidImage, image_pixels
from .registry import file_hash

# Pinned download (exact commit) and checksum, so the same model file is always used.
MODEL_URL = (
    "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/"
    "d3e5499fa8701ff0453ca940a8dfeae39b2f1504/onnx/model_quantized.onnx"
)
MODEL_SHA256 = "9a98d6daf3d926869ab8cc4c2ed7374a2bc23b889bb7ca3b0915d15e3c4756bb"
# ImageNet mean and standard deviation per RGB channel, which SegFormer expects.
IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# Zero-based ADE20K labels that count as a drivable surface: road, sidewalk, path, dirt track
# and earth/ground. Gravel and dirt roads are very often labelled "earth" by the model, so
# leaving it out made the check reject many valid unpaved roads.
SURFACE_LABELS = (6, 11, 52, 91, 13)
# At least this fraction of the lower-centre region must be confident surface pixels.
MIN_ROAD_COVERAGE = 0.18


class RoadPhotoCheckUnavailable(RuntimeError):
    """The road-photo check could not safely evaluate this image."""


def model_path():
    """Where the ONNX segmentation model is stored (can be overridden with an env var)."""
    default = Path(__file__).resolve().parents[1] / "models" / "road-photo-segformer.onnx"
    return Path(os.getenv("ROADSENSE_ROAD_PHOTO_MODEL", default))


def road_coverage(logits):
    """Measure confident surface pixels in the classifier's lower-center view.

    `logits` has shape (150, H, W): one score map for each of the 150 ADE20K classes.
    Returns the fraction (0..1) of pixels in the focal region that are confidently road-like.
    """
    if logits.ndim != 3 or logits.shape[0] != 150 or min(logits.shape[1:]) < 4:
        raise ValueError("Unexpected road-photo model output shape.")
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite road-photo model output.")
    _, height, width = logits.shape
    # Same lower-centre box as the classifier's focal crop: bottom half, middle half of the width.
    focal = logits[:, height // 2:, width // 4:3 * width // 4]
    # For every pixel, the class with the highest score.
    winners = focal.argmax(axis=0)
    # Softmax probability of the winning class at each pixel.
    # softmax(max) = exp(max) / sum(exp(all)) = 1 / sum(exp(all - max)); subtracting max avoids overflow.
    confidence = 1 / np.exp(focal - focal.max(axis=0)).sum(axis=0)
    # A pixel counts only if its winning class is a road-like surface AND the model is at least 45% sure.
    return float(np.mean(np.isin(winners, SURFACE_LABELS) & (confidence >= 0.45)))


class RoadPhotoValidator:
    """Callable object: `validator(image)` raises InvalidImage if no road is visible."""

    def __init__(self, path):
        self.path = path
        self._session = None   # the ONNX model is loaded lazily on first use
        self._lock = Lock()

    def _load(self):
        """Load the ONNX model once (thread-safe) after verifying its checksum."""
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
            # Resize to 512x512, scale to 0..1, then normalise with ImageNet mean/std.
            pixels = (image_pixels(image, (512, 512)) / 255 - IMAGE_MEAN) / IMAGE_STD
            # ONNX/PyTorch models want (batch, channels, height, width):
            # transpose(2, 0, 1) turns (H, W, C) into (C, H, W); [None] adds the batch axis.
            inputs = np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])
            # [0][0]: first output ("logits"), first image in the batch -> shape (150, H, W).
            logits = self._load().run(["logits"], {"pixel_values": inputs})[0][0]
            coverage = road_coverage(logits)
        except Exception as exc:
            raise RoadPhotoCheckUnavailable("The road-photo check is temporarily unavailable. Try again later.") from exc
        # Reject the photo if too little of the focal region is confident road surface.
        if coverage < MIN_ROAD_COVERAGE:
            raise InvalidImage("Invalid road photo. Choose an image with a clear road surface in the lower center.")


def install_model(path):
    """Download the SegFormer model if it is missing, verifying its checksum before using it."""
    if path.is_file() and file_hash(path) == MODEL_SHA256:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Download into a temporary file first, then move it into place only if the checksum
    # matches. This way an interrupted download never leaves a broken model file behind.
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


# Run `python -m backend.app.road_photo` to download the model.
if __name__ == "__main__":
    path = model_path()
    install_model(path)
    print(f"Road-photo model ready at {path}")
