import pytest
import torch
from PIL import Image

from config import XRAY_CLASSIFIER_CHECKPOINT
from services import xray_gate_model
from services.modality_validation import SuitabilityServiceUnavailable, SuitabilityStatus


def test_training_preprocessing_matches_supplied_rgb_resize_and_normalization():
    image = Image.new("RGB", (32, 16), (255, 0, 0))
    tensor = xray_gate_model.preprocess_for_classifier(image)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    expected = torch.tensor(
        [
            (1.0 - 0.485) / 0.229,
            (0.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ]
    )
    torch.testing.assert_close(tensor[0, :, 100, 100], expected)


@pytest.mark.parametrize(
    ("score", "threshold", "expected"),
    [
        (0.8, 0.8, SuitabilityStatus.ACCEPTED),
        (0.2, 0.5, SuitabilityStatus.REJECTED),
        (0.5, 0.5, SuitabilityStatus.ACCEPTED),
        (0.6, 0.7, SuitabilityStatus.REJECTED),
    ],
)
def test_configurable_decision_boundary(score, threshold, expected, monkeypatch):
    monkeypatch.setattr(xray_gate_model, "predict_xray_score", lambda _image: score)
    monkeypatch.setattr(xray_gate_model, "XRAY_CLASSIFIER_THRESHOLD", threshold)

    decision = xray_gate_model.classify_chest_xray(Image.new("RGB", (12, 12)))

    assert decision.status is expected
    assert decision.xray_score == score
    assert decision.classifier_id == xray_gate_model.XRAY_CLASSIFIER_ID


@pytest.mark.skipif(
    not XRAY_CLASSIFIER_CHECKPOINT.is_file(),
    reason="User-provided xray_classifier.pth is not present in this checkout.",
)
def test_provided_checkpoint_loads_strictly_and_scores_sample_xray():
    path, modified_ns, file_size = xray_gate_model._checkpoint_signature()
    model, _torch, classes = xray_gate_model._load_model(path, modified_ns, file_size)
    assert model.__class__.__name__ == "MobileNetV2"
    assert list(classes) == ["non_xray", "xray"]
    assert model.training is False

    image = Image.open("assets/Xray sample.jpg")
    score = xray_gate_model.predict_xray_score(image)
    assert 0.0 <= score <= 1.0
    decision = xray_gate_model.classify_chest_xray(image)
    assert decision.xray_score == pytest.approx(score)
    assert decision.status is (
        SuitabilityStatus.ACCEPTED
        if score >= xray_gate_model.XRAY_CLASSIFIER_THRESHOLD
        else SuitabilityStatus.REJECTED
    )


def test_missing_checkpoint_is_reported_without_loading_weights(tmp_path, monkeypatch):
    missing = tmp_path / "missing.pth"
    monkeypatch.setattr(xray_gate_model, "XRAY_CLASSIFIER_CHECKPOINT", missing)
    path, modified_ns, file_size = xray_gate_model._checkpoint_signature()
    with pytest.raises(SuitabilityServiceUnavailable, match="validation model not found"):
        xray_gate_model._load_model(path, modified_ns, file_size)


def test_checkpoint_or_threshold_change_updates_classifier_revision(tmp_path, monkeypatch):
    checkpoint = tmp_path / "xray_classifier.pth"
    checkpoint.write_bytes(b"first")
    monkeypatch.setattr(xray_gate_model, "XRAY_CLASSIFIER_CHECKPOINT", checkpoint)
    first_revision = xray_gate_model.classifier_revision()

    checkpoint.write_bytes(b"second checkpoint")
    second_revision = xray_gate_model.classifier_revision()
    assert second_revision != first_revision

    monkeypatch.setattr(xray_gate_model, "XRAY_CLASSIFIER_THRESHOLD", 0.7)
    assert xray_gate_model.classifier_revision() != second_revision


def test_xray_score_uses_checkpoint_class_name_index_from_softmax(monkeypatch):
    class MockModel:
        def to(self, _device):
            return self

        def __call__(self, _tensor):
            return torch.tensor([[torch.log(torch.tensor(4.0)), 0.0]])

    monkeypatch.setattr(xray_gate_model, "_checkpoint_signature", lambda: ("checkpoint", 1, 1))
    monkeypatch.setattr(
        xray_gate_model,
        "_load_model",
        lambda *_args: (MockModel(), torch, ["xray", "non_xray"]),
    )
    monkeypatch.setattr(
        xray_gate_model,
        "preprocess_for_classifier",
        lambda _image: torch.zeros((1, 3, 224, 224)),
    )

    score = xray_gate_model.predict_xray_score(Image.new("RGB", (16, 16)))

    assert score == pytest.approx(0.8)


@pytest.mark.skipif(
    not XRAY_CLASSIFIER_CHECKPOINT.is_file(),
    reason="User-provided xray_classifier.pth is not present in this checkout.",
)
def test_provided_checkpoint_rejects_a_photo_like_rgb_image():
    image = Image.new("RGB", (224, 224), (180, 110, 75))
    decision = xray_gate_model.classify_chest_xray(image)
    assert decision.status is SuitabilityStatus.REJECTED
    assert decision.xray_score is not None
    assert decision.xray_score < xray_gate_model.XRAY_CLASSIFIER_THRESHOLD
