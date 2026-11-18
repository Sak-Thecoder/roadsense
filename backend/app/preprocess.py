"""Shared pixel decoding and model preprocessing for training and inference."""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MODELS = ("mobilenetv2", "resnet50", "efficientnetb0")
CLASSES = ("Good", "Moderate", "Poor")
INPUT_SIZE = (224, 224)
PREPROCESSING = "rgb-exif-bilinear224-keras-v1"
# Bottom-centre road region cropped from the decoded full-resolution image before resizing.
FOCAL_PREPROCESSING = "rgb-exif-focalcrop-bilinear224-keras-v1"
PREPROCESSING_MODES = (PREPROCESSING, FOCAL_PREPROCESSING)
FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImage(ValueError):
    """Invalid or unsupported image input."""


def decode(data: bytes, max_pixels: int = 24_000_000, focal: bool = False) -> np.ndarray:
    return image_pixels(decode_image(data, max_pixels), focal=focal)


def focal_region(image: Image.Image) -> Image.Image:
    """Central half of the width and lower half of the height, where the road surface usually is."""
    width, height = image.size
    return image.crop((round(width * 0.25), round(height * 0.5), round(width * 0.75), height))


def image_pixels(image: Image.Image, size=INPUT_SIZE, focal: bool = False) -> np.ndarray:
    if focal:
        image = focal_region(image)
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def uses_focal_crop(preprocessing: str) -> bool:
    if preprocessing not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing {preprocessing!r}")
    return preprocessing == FOCAL_PREPROCESSING


def decode_image(data: bytes, max_pixels: int = 24_000_000) -> Image.Image:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in FORMATS or getattr(image, "n_frames", 1) != 1:
                raise InvalidImage("Choose a single JPEG, PNG or WebP image.")
            if image.width * image.height > max_pixels:
                raise InvalidImage("The image dimensions exceed the supported limit.")
            return ImageOps.exif_transpose(image).convert("RGB")
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
