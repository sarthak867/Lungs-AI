from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from services.preprocessing import (
    FileValidationCode,
    ImageValidationError,
    validate_image,
    validate_uploaded_file,
)


def png_bytes(size=(48, 32)):
    output = BytesIO()
    Image.new("L", size, color=127).save(output, format="PNG")
    return output.getvalue()


def test_validate_image_accepts_png_and_reports_dimensions():
    validated = validate_image(png_bytes(), "sample.png")
    assert (validated.width, validated.height) == (48, 32)
    assert validated.image.mode == "RGB"


def test_validate_image_applies_exif_orientation():
    output = BytesIO()
    image = Image.new("RGB", (48, 32), color="white")
    exif = image.getexif()
    exif[274] = 6
    image.save(output, format="JPEG", exif=exif)
    validated = validate_image(output.getvalue(), "oriented.jpg")
    assert (validated.width, validated.height) == (32, 48)
    assert validated.image.mode == "RGB"


def test_file_validation_preserves_rgb_photo_colors_for_suitability_gate():
    output = BytesIO()
    Image.new("RGB", (8, 8), color=(220, 40, 10)).save(output, format="PNG")

    validated = validate_image(output.getvalue(), "photo.png")

    assert validated.image.mode == "RGB"
    assert validated.image.getpixel((0, 0)) == (220, 40, 10)


def test_file_validation_normalizes_16_bit_png_before_rgb_conversion():
    output = BytesIO()
    image = Image.fromarray(np.array([[0, 65535]], dtype=np.uint16))
    image.save(output, format="PNG")

    validated = validate_image(output.getvalue(), "16-bit.png")
    pixels = np.asarray(validated.image)

    assert validated.image.mode == "RGB"
    assert pixels[0, 0, 0] == 0
    assert pixels[0, 1, 0] == 255


@pytest.mark.parametrize(
    ("data", "filename"),
    [
        (b"not an image", "sample.jpg"),
        (png_bytes(), "sample.gif"),
        (b"", "sample.png"),
    ],
)
def test_validate_image_rejects_invalid_inputs(data, filename):
    with pytest.raises(ImageValidationError):
        validate_image(data, filename)


def test_file_validation_returns_structured_result_for_corrupt_bytes():
    result = validate_uploaded_file(b"corrupt bytes", "sample.png")
    assert not result.accepted
    assert result.code is FileValidationCode.CORRUPT_IMAGE
    assert result.validated_image is None
    assert result.reason


def test_file_validation_returns_structured_result_for_unsupported_extension():
    result = validate_uploaded_file(png_bytes(), "sample.webp")
    assert not result.accepted
    assert result.code is FileValidationCode.UNSUPPORTED_FORMAT


def test_file_validation_rejects_unsupported_content_even_with_png_extension():
    output = BytesIO()
    Image.new("RGB", (8, 8), color="blue").save(output, format="WEBP")
    result = validate_uploaded_file(output.getvalue(), "misnamed.png")
    assert not result.accepted
    assert result.code is FileValidationCode.UNSUPPORTED_FORMAT


def test_validate_image_rejects_excessive_dimensions(monkeypatch):
    monkeypatch.setattr("services.preprocessing.MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ImageValidationError, match="dimensions"):
        validate_image(png_bytes(), "sample.png")


def test_validate_image_rejects_excessive_file_size(monkeypatch):
    data = png_bytes()
    monkeypatch.setattr("services.preprocessing.MAX_UPLOAD_BYTES", len(data) - 1)
    with pytest.raises(ImageValidationError, match="10 MB"):
        validate_image(data, "sample.png")
