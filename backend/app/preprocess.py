"""Shared pixel decoding and model preprocessing for training and inference.

Both the training scripts and the web API import this file, so a photo is turned
into numbers in exactly the same way when the model learns and when it predicts.
If the two ever differed, the model would see "different-looking" images in
production and its accuracy would drop.
"""
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# The three CNN architectures we compare, and the three road-quality grades they predict.
MODELS = ("mobilenetv2", "resnet50", "efficientnetb0")
CLASSES = ("Good", "Moderate", "Poor")
# All three backbones were pre-trained on ImageNet at 224x224 pixels.
INPUT_SIZE = (224, 224)
# Preprocessing IDs are stored in each model's .json file. The API reads the ID to
# know which steps the model was trained with (full photo vs. cropped road region).
PREPROCESSING = "rgb-exif-bilinear224-keras-v1"
# Bottom-centre road region cropped from the decoded full-resolution image before resizing.
FOCAL_PREPROCESSING = "rgb-exif-focalcrop-bilinear224-keras-v1"
PREPROCESSING_MODES = (PREPROCESSING, FOCAL_PREPROCESSING)
FORMATS = {"JPEG", "PNG", "WEBP"}


class InvalidImage(ValueError):
    """Invalid or unsupported image input."""


def decode(data: bytes, max_pixels: int = 24_000_000, focal: bool = False) -> np.ndarray:
    """Convenience helper: raw file bytes -> 224x224x3 float array in one call."""
    return image_pixels(decode_image(data, max_pixels), focal=focal)


def focal_region(image: Image.Image) -> Image.Image:
    """Central half of the width and lower half of the height, where the road surface usually is.

    Dashboard/street photos contain a lot of sky, buildings and cars. Cropping to
    the lower-centre box keeps the model focused on the asphalt in front of the
    camera. The crop is taken from the full-resolution photo, so cracks and
    potholes keep their detail before the image is shrunk to 224x224.
    """
    width, height = image.size
    # PIL crop box is (left, top, right, bottom):
    # left = 25% of width, top = 50% of height, right = 75% of width, bottom = full height.
    return image.crop((round(width * 0.25), round(height * 0.5), round(width * 0.75), height))


def image_pixels(image: Image.Image, size=INPUT_SIZE, focal: bool = False) -> np.ndarray:
    """Optionally crop to the road region, then resize and return an (H, W, 3) float32 array (0-255)."""
    if focal:
        image = focal_region(image)
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def uses_focal_crop(preprocessing: str) -> bool:
    """Return True if a model's stored preprocessing ID means "crop to the road region first"."""
    if preprocessing not in PREPROCESSING_MODES:
        raise ValueError(f"Unknown preprocessing {preprocessing!r}")
    return preprocessing == FOCAL_PREPROCESSING


def decode_image(data: bytes, max_pixels: int = 24_000_000) -> Image.Image:
    """Safely open uploaded bytes as an upright RGB PIL image, or raise InvalidImage."""
    try:
        with Image.open(BytesIO(data)) as image:
            # Reject unsupported formats and animated images (GIF/WebP with several frames).
            if image.format not in FORMATS or getattr(image, "n_frames", 1) != 1:
                raise InvalidImage("Choose a single JPEG, PNG or WebP image.")
            # Guard against huge images that would use too much memory (a "decompression bomb").
            if image.width * image.height > max_pixels:
                raise InvalidImage("The image dimensions exceed the supported limit.")
            # Phones store rotation in EXIF metadata instead of rotating the pixels.
            # exif_transpose applies that rotation so the road is the right way up;
            # convert("RGB") removes transparency / greyscale so we always get 3 channels.
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InvalidImage("The image could not be decoded. Choose a valid JPEG, PNG or WebP.") from exc


def preprocess(pixels: np.ndarray, model: str) -> np.ndarray:
    """Keras Applications transforms, with EfficientNet scaling inside its backbone.

    Each pre-trained network expects pixel values in the range it was trained on
    with ImageNet, so we apply that network's own normalisation here.
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    if model == "mobilenetv2":
        # Scale 0..255 to -1..1.
        return pixels / 127.5 - 1.0
    if model == "resnet50":
        # "Caffe" style: reverse RGB -> BGR (pixels[..., ::-1] flips the last axis),
        # then subtract the mean ImageNet colour of each channel.
        return pixels[..., ::-1] - np.array([103.939, 116.779, 123.68], dtype=np.float32)
    if model == "efficientnetb0":
        # EfficientNet has its own rescaling layer built in, so it takes raw 0..255 values.
        return pixels
    raise ValueError("Unknown model")
