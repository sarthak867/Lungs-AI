from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, MODEL_INPUT_SIZE

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a supported, safe image."""


class FileValidationCode(str, Enum):
    VALID = "valid"
    UNSUPPORTED_FORMAT = "unsupported_format"
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    IMAGE_TOO_LARGE = "image_too_large"
    CORRUPT_IMAGE = "corrupt_image"


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image
    width: int
    height: int


@dataclass(frozen=True)
class FileValidationResult:
    accepted: bool
    code: FileValidationCode
    reason: str
    validated_image: ValidatedImage | None = None


def validate_uploaded_file(data: bytes, filename: str) -> FileValidationResult:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if f".{suffix}" not in ALLOWED_EXTENSIONS:
        return FileValidationResult(
            False, FileValidationCode.UNSUPPORTED_FORMAT, "Choose a PNG, JPG, or JPEG image."
        )
    if not data:
        return FileValidationResult(False, FileValidationCode.EMPTY_FILE, "The selected file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        return FileValidationResult(
            False, FileValidationCode.FILE_TOO_LARGE, "The image is larger than the 10 MB upload limit."
        )

    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG"}:
                return FileValidationResult(
                    False,
                    FileValidationCode.UNSUPPORTED_FORMAT,
                    "The file contents are not a PNG or JPEG image.",
                )
            if source.width * source.height > MAX_IMAGE_PIXELS:
                return FileValidationResult(
                    False,
                    FileValidationCode.IMAGE_TOO_LARGE,
                    "The image dimensions exceed the supported limit.",
                )
            source.verify()
        with Image.open(BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source)
            if image.mode in {"I;16", "I", "F"}:
                pixels = np.asarray(image).astype(np.float32)
                pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min() + 1e-8)
                image = Image.fromarray((pixels * 255).astype(np.uint8))
            image = image.convert("RGB")
            image.load()
            setattr(image, "_raw_bytes", data)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        return FileValidationResult(
            False,
            FileValidationCode.CORRUPT_IMAGE,
            "This image could not be read. Try another PNG or JPEG file.",
        )

    validated_image = ValidatedImage(image=image, width=image.width, height=image.height)
    return FileValidationResult(
        True,
        FileValidationCode.VALID,
        "The image file is readable and meets the upload limits.",
        validated_image,
    )


def validate_image(data: bytes, filename: str) -> ValidatedImage:
    """Compatibility helper for callers that require a readable image or an exception."""
    result = validate_uploaded_file(data, filename)
    if not result.accepted or result.validated_image is None:
        raise ImageValidationError(result.reason)
    return result.validated_image


def preprocess_image(image: Image.Image, input_size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Prepare a grayscale tensor using TorchXRayVision's center-crop and normalization."""
    if input_size <= 0:
        raise ValueError("input_size must be greater than zero.")
    resized = center_crop_image(image).resize((input_size, input_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.float32)
    normalized = (2.0 * (pixels / 255.0) - 1.0) * 1024.0
    return normalized[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False)


def center_crop_image(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    side = min(grayscale.size)
    left = (grayscale.width - side) // 2
    top = (grayscale.height - side) // 2
    return grayscale.crop((left, top, left + side, top + side))
