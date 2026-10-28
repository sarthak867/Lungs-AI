import pytest
import torch
from PIL import Image

from config import DISEASE_MODEL_CHECKPOINT
from services.inference import (
    PATHOLOGY_LABELS,
    ModelUnavailableError,
    _file_signature,
    _load_model,
    load_thresholds,
    preprocess_disease_image,
    run_inference,
)
from services.modality_validation import (
    SuitabilityDecision,
    SuitabilityGateError,
    SuitabilityStatus,
    validate_chest_xray_suitability,
)


@pytest.mark.parametrize(
    "status",
    [
        SuitabilityStatus.REJECTED,
        SuitabilityStatus.UNABLE_TO_VERIFY,
        SuitabilityStatus.SERVICE_UNAVAILABLE,
    ],
)
def test_disease_inference_is_never_called_without_suitability_pass(monkeypatch, status):
    def fail_if_called():
        pytest.fail("Disease model must not load for a non-accepted X-ray gate result")

    monkeypatch.setattr("services.inference.load_model", fail_if_called)
    image = Image.new("RGB", (16, 16))
    rejected = validate_chest_xray_suitability(
        image,
        "image-hash",
        lambda _image: SuitabilityDecision(status, "mock non-pass"),
    )
    with pytest.raises(SuitabilityGateError):
        run_inference(image, image_hash="image-hash", validation=rejected)


def test_disease_model_is_strictly_loaded_and_in_evaluation_mode():
    checkpoint_path, modified_ns, file_size = _file_signature(DISEASE_MODEL_CHECKPOINT)
    model, _torch = _load_model(checkpoint_path, modified_ns, file_size)

    assert model.__class__.__name__ == "DenseNet"
    assert model.classifier.out_features == 14
    assert model.training is False


def test_disease_preprocessing_matches_training_transform():
    image = Image.new("RGB", (20, 10), (255, 0, 0))
    tensor = preprocess_disease_image(image)

    assert tensor.shape == (1, 3, 320, 320)
    assert tensor.dtype == torch.float32
    expected_gray = 0.299
    expected = torch.tensor(
        [
            (expected_gray - 0.485) / 0.229,
            (expected_gray - 0.456) / 0.224,
            (expected_gray - 0.406) / 0.225,
        ]
    )
    torch.testing.assert_close(tensor[0, :, 100, 100], expected, atol=0.01, rtol=0.01)


def test_disease_inference_uses_sigmoid_and_per_label_thresholds(monkeypatch):
    class MockModel:
        def to(self, _device):
            return self

        def __call__(self, _tensor):
            logits = torch.full((1, len(PATHOLOGY_LABELS)), -2.0)
            logits[0, 0] = 2.0
            return logits

    image = Image.new("RGB", (16, 16))
    validation = validate_chest_xray_suitability(
        image,
        "image-hash",
        lambda _image: SuitabilityDecision(
            SuitabilityStatus.ACCEPTED,
            "Mock gate pass for inference application logic test.",
        ),
    )
    monkeypatch.setattr("services.inference.load_model", lambda: (MockModel(), torch))
    monkeypatch.setattr(
        "services.inference.load_thresholds",
        lambda: {label: 0.5 for label in PATHOLOGY_LABELS},
    )
    monkeypatch.setattr("services.inference.disease_model_revision", lambda: "test-revision")
    monkeypatch.setattr(
        "services.inference.preprocess_disease_image",
        lambda _image: torch.zeros((1, 3, 320, 320)),
    )

    result = run_inference(image, image_hash="image-hash", validation=validation)

    assert len(result.predictions) == 14
    assert result.pathologies == PATHOLOGY_LABELS
    assert result.predictions[0].score == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    assert result.predictions[0].meets_threshold
    assert len(result.findings) == 1
    assert result.model_revision == "test-revision"


def test_no_threshold_crossings_are_not_reported_as_normal(monkeypatch):
    class MockModel:
        def to(self, _device):
            return self

        def __call__(self, _tensor):
            return torch.full((1, len(PATHOLOGY_LABELS)), -8.0)

    image = Image.new("RGB", (16, 16))
    validation = validate_chest_xray_suitability(
        image,
        "image-hash",
        lambda _image: SuitabilityDecision(SuitabilityStatus.ACCEPTED, "Mock gate pass."),
    )
    monkeypatch.setattr("services.inference.load_model", lambda: (MockModel(), torch))
    monkeypatch.setattr(
        "services.inference.load_thresholds",
        lambda: {label: 0.5 for label in PATHOLOGY_LABELS},
    )
    monkeypatch.setattr("services.inference.disease_model_revision", lambda: "test-revision")
    monkeypatch.setattr(
        "services.inference.preprocess_disease_image",
        lambda _image: torch.zeros((1, 3, 320, 320)),
    )

    result = run_inference(image, image_hash="image-hash", validation=validation)

    assert result.findings == ()
    assert all(not prediction.meets_threshold for prediction in result.predictions)


def test_threshold_findings_are_ordered_by_score_descending():
    from services.inference import AnalysisResult, Prediction

    result = AnalysisResult(
        model_name="Mock model (unit test only)",
        checkpoint_id="test-checkpoint",
        timestamp="test timestamp",
        predictions=(
            Prediction("Atelectasis", 0.6, 0.5),
            Prediction("Cardiomegaly", 0.9, 0.5),
            Prediction("Consolidation", 0.4, 0.5),
        ),
        score_semantics="Mock test scores",
        raw_scores=(0.0, 0.0, 0.0),
        pathologies=("Atelectasis", "Cardiomegaly", "Consolidation"),
    )

    assert [finding.label for finding in result.findings] == ["Cardiomegaly", "Atelectasis"]


def test_checkpoint_smoke_inference_returns_14_finite_sigmoid_scores():
    if not DISEASE_MODEL_CHECKPOINT.is_file():
        pytest.skip("Project-provided disease checkpoint is not present.")
    image = Image.open("assets/Xray sample.jpg")
    validation = validate_chest_xray_suitability(
        image,
        "checkpoint-smoke-test",
        lambda _image: SuitabilityDecision(SuitabilityStatus.ACCEPTED, "Mock gate pass for checkpoint smoke test."),
    )

    result = run_inference(image, image_hash="checkpoint-smoke-test", validation=validation)

    assert len(result.predictions) == 14
    assert all(0.0 <= prediction.score <= 1.0 for prediction in result.predictions)
    assert all(prediction.threshold is not None for prediction in result.predictions)
    assert result.checkpoint_id == DISEASE_MODEL_CHECKPOINT.name


def test_threshold_file_has_exact_labels_and_usable_cutoffs():
    thresholds = load_thresholds()

    assert tuple(thresholds) == PATHOLOGY_LABELS
    assert all(0.0 <= threshold <= 1.0 for threshold in thresholds.values())


def test_missing_disease_checkpoint_is_reported(tmp_path):
    with pytest.raises(ModelUnavailableError, match="not found"):
        _load_model(str(tmp_path / "missing.pt"), 0, 0)
