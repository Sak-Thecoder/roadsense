"""Shared pixel decoding and model preprocessing for training and inference."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MODELS = ("mobilenetv2", "resnet50", "efficientnetb0")
CLASSES = ("Good", "Moderate", "Poor")
INPUT_SIZE = (224, 224)
PREPROCESSING = "rgb-exif-bilinear224-keras-v1"
FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImage(ValueError):
    """Invalid or unsupported image input."""


def decode(data: bytes, max_pixels: int = 24_000_000) -> np.ndarray:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in FORMATS or getattr(image, "n_frames", 1) != 1:
                raise InvalidImage("Choose a single JPEG, PNG or WebP image.")
            if image.width * image.height > max_pixels:
                raise InvalidImage("The image dimensions exceed the supported limit.")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            return np.asarray(rgb.resize(INPUT_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImage("The image could not be decoded. Choose a valid JPEG, PNG or WebP.") from exc


def preprocess(pixels: np.ndarray, model: str) -> np.ndarray:
    """Keras Applications transforms, with EfficientNet scaling inside its backbone."""
    pixels = np.asarray(pixels, dtype=np.float32)
    if model == "mobilenetv2":
        return pixels / 127.5 - 1.0
    if model == "resnet50":
        return pixels[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype=np.float32)
    if model == "efficientnetb0":
        return pixels
    raise ValueError("Unknown model")
