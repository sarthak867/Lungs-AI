from hashlib import sha256
import pytest
from PIL import Image, ImageDraw

from services.modality_validation import (
    SuitabilityDecision,
    SuitabilityGateError,
    SuitabilityServiceUnavailable,
    SuitabilityStatus,
    SuitabilityValidationResult,
    fingerprint_image,
    require_suitability_pass,
    suitability_message,
    validate_chest_xray_suitability,
)

TEST_HASH = sha256(b"unit-test-image").hexdigest()


def make_image(color, mode="RGB"):
    return Image.new(mode, (64, 64), color)


def test_rgb_photo_rejection_path_uses_mock_classifier_not_pixel_heuristics():
    photo = make_image((135, 206, 235))
    draw = ImageDraw.Draw(photo)
    draw.rectangle((0, 42, 63, 63), fill=(34, 139, 34))
    draw.ellipse((44, 6, 56, 18), fill=(255, 255, 0))
    draw.rectangle((14, 28, 43, 53), fill=(160, 82, 45))
    draw.polygon(((11, 30), (28, 17), (46, 30)), fill=(178, 34, 34))
    called = []

    def mocked_classifier(image):
        called.append(image)
        return SuitabilityDecision(SuitabilityStatus.REJECTED, "Mock rejection for gate test.")

    result = validate_chest_xray_suitability(photo, TEST_HASH, mocked_classifier)
    assert result.status is SuitabilityStatus.REJECTED
    assert called == [photo]


def test_synthetic_xray_like_image_pass_path_uses_mock_classifier_only():
    synthetic_image = make_image(48, mode="L")
    draw = ImageDraw.Draw(synthetic_image)
    draw.line((32, 2, 32, 25), fill=210, width=3)
    draw.ellipse((8, 13, 29, 56), outline=180, width=2)
    draw.ellipse((35, 13, 56, 56), outline=180, width=2)
    for y in (22, 30, 38, 46):
        draw.arc((5, y - 8, 32, y + 10), start=200, end=340, fill=130, width=2)
        draw.arc((32, y - 8, 59, y + 10), start=200, end=340, fill=130, width=2)
    result = validate_chest_xray_suitability(
        synthetic_image,
        TEST_HASH,
        lambda _image: SuitabilityDecision(SuitabilityStatus.ACCEPTED, "Mock pass for gate test."),
    )
    assert result.status is SuitabilityStatus.ACCEPTED
    assert result.content_hash == TEST_HASH


def test_classifier_error_routes_to_unable_to_verify():
    def broken_classifier(_image):
        raise RuntimeError("mock classifier error")

    result = validate_chest_xray_suitability(make_image("white"), TEST_HASH, broken_classifier)
    assert result.status is SuitabilityStatus.UNABLE_TO_VERIFY


def test_explicit_classifier_unavailable_state_is_preserved():
    def unavailable_classifier(_image):
        raise SuitabilityServiceUnavailable("weights are not installed")

    result = validate_chest_xray_suitability(make_image("white"), TEST_HASH, unavailable_classifier)
    assert result.status is SuitabilityStatus.SERVICE_UNAVAILABLE
    assert result.reason == "weights are not installed"


def test_missing_local_checkpoint_returns_service_unavailable(monkeypatch):
    from services import xray_gate_model

    monkeypatch.setattr(
        xray_gate_model,
        "XRAY_CLASSIFIER_CHECKPOINT",
        xray_gate_model.XRAY_CLASSIFIER_CHECKPOINT.with_name("missing-test-checkpoint.pth"),
    )
    result = validate_chest_xray_suitability(make_image("white"), TEST_HASH)
    assert result.status is SuitabilityStatus.SERVICE_UNAVAILABLE
    assert not result.accepted


def test_user_messages_distinguish_all_suitability_states():
    expected_messages = {
        SuitabilityStatus.ACCEPTED: "Chest X-ray input accepted for research analysis",
        SuitabilityStatus.REJECTED: "Image rejected: this does not appear to be a supported chest X-ray",
        SuitabilityStatus.UNABLE_TO_VERIFY: "Unable to verify image type; analysis blocked",
        SuitabilityStatus.SERVICE_UNAVAILABLE: "Image validation service unavailable",
    }
    for status, expected in expected_messages.items():
        result = SuitabilityValidationResult(status, TEST_HASH, "test reason")
        assert suitability_message(result).startswith(expected)


def test_default_validator_uses_local_classifier(monkeypatch):
    from services import xray_gate_model

    observed = []

    def predict(_image):
        observed.append("called")
        return 0.91

    monkeypatch.setattr(xray_gate_model, "predict_xray_score", predict)
    result = validate_chest_xray_suitability(make_image("white"), TEST_HASH)
    assert observed == ["called"]
    assert result.status is SuitabilityStatus.ACCEPTED
    assert result.classifier_id == xray_gate_model.XRAY_CLASSIFIER_ID
    assert result.xray_score == 0.91


def test_matching_pass_is_accepted_but_stale_or_rejected_results_are_blocked():
    image = make_image("white")
    accepted = SuitabilityValidationResult(
        SuitabilityStatus.ACCEPTED,
        TEST_HASH,
        "mock accepted",
        image_fingerprint=fingerprint_image(image),
    )
    require_suitability_pass(accepted, TEST_HASH, image)
    with pytest.raises(SuitabilityGateError):
        require_suitability_pass(accepted, "different-image-hash", image)
    rejected = SuitabilityValidationResult(SuitabilityStatus.REJECTED, TEST_HASH, "mock rejected")
    with pytest.raises(SuitabilityGateError):
        require_suitability_pass(rejected, TEST_HASH, image)
    with pytest.raises(SuitabilityGateError):
        require_suitability_pass(accepted, TEST_HASH, make_image("black"))
