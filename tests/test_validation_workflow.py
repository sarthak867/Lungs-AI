from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from services.explainability import generate_gradcam
from services.inference import AnalysisResult, Prediction
from services.modality_validation import (
    SuitabilityDecision,
    SuitabilityGateError,
    SuitabilityStatus,
    SuitabilityValidationResult,
    fingerprint_image,
    validate_chest_xray_suitability,
)
from services.preprocessing import validate_image
from services.upload_state import (
    attach_suitability_result,
    register_invalid_upload,
    register_valid_upload,
)


def png_bytes(color=120):
    output = BytesIO()
    Image.new("L", (40, 40), color=color).save(output, format="PNG")
    return output.getvalue()


def test_changing_upload_clears_previous_predictions_heatmaps_and_report():
    state = {
        "analysis": object(),
        "analysis_hash": "old-hash",
        "gradcam": object(),
        "gradcams": {"old label": object()},
        "gradcam_errors": {"old label": "old error"},
        "gradcam_label": "old label",
        "report_bytes": b"old report",
    }
    first = png_bytes(80)
    register_valid_upload(state, first, "first.png", validate_image(first, "first.png"))
    accepted = validate_chest_xray_suitability(
        validate_image(first, "first.png").image,
        state["image_hash"],
        lambda _image: SuitabilityDecision(
            SuitabilityStatus.ACCEPTED,
            "Mock pass for state-reset test only.",
        ),
    )
    assert attach_suitability_result(state, accepted)

    second = png_bytes(180)
    register_valid_upload(state, second, "first.png", validate_image(second, "first.png"))
    assert "analysis" not in state
    assert "gradcam" not in state
    assert "gradcams" not in state
    assert "gradcam_errors" not in state
    assert "report_bytes" not in state
    assert "suitability_validation" not in state
    assert state["image_hash"] != accepted.content_hash


def test_stale_validation_result_cannot_be_attached_to_current_upload():
    data = png_bytes()
    state = {}
    register_valid_upload(state, data, "sample.png", validate_image(data, "sample.png"))
    stale = SuitabilityValidationResult(
        SuitabilityStatus.ACCEPTED,
        "not-the-current-image",
        "Mock stale result.",
    )
    assert not attach_suitability_result(state, stale)
    assert "suitability_validation" not in state


def test_failed_suitability_result_clears_any_previous_analysis_artifacts():
    data = png_bytes()
    image = validate_image(data, "sample.png").image
    state = {}
    register_valid_upload(state, data, "sample.png", validate_image(data, "sample.png"))
    state.update(
        analysis=object(),
        analysis_hash=state["image_hash"],
        gradcam=object(),
        report_bytes=b"previous report",
    )
    rejected = validate_chest_xray_suitability(
        image,
        state["image_hash"],
        lambda _image: SuitabilityDecision(SuitabilityStatus.REJECTED, "Mock rejection."),
    )
    assert attach_suitability_result(state, rejected)
    assert "analysis" not in state
    assert "gradcam" not in state
    assert "report_bytes" not in state


def test_invalid_file_clears_previous_upload_and_all_analysis_artifacts():
    state = {
        "image_bytes": b"old file",
        "image_hash": "old hash",
        "image_name": "old.png",
        "suitability_validation": object(),
        "analysis": object(),
        "gradcam": object(),
        "report_bytes": b"old report",
    }
    register_invalid_upload(state, "corrupt image")
    assert not any(
        key in state
        for key in (
            "image_bytes",
            "image_hash",
            "suitability_validation",
            "analysis",
            "gradcam",
            "report_bytes",
        )
    )
    assert state["file_validation_error"] == "corrupt image"


def test_gradcam_is_blocked_before_model_access_when_gate_fails(monkeypatch):
    def fail_if_model_loaded():
        pytest.fail("Grad-CAM must not access the model before suitability passes")

    monkeypatch.setattr("services.explainability.load_model", fail_if_model_loaded)
    image = Image.new("L", (32, 32))
    result = AnalysisResult(
        model_name="Mock model (unit test only)",
        checkpoint_id="mock-checkpoint",
        timestamp="test timestamp",
        predictions=(Prediction("Mock label", 0.1),),
        score_semantics="Mock score",
        raw_scores=(0.1,),
        pathologies=("Mock label",),
        image_hash="rejected-image",
        image_fingerprint=fingerprint_image(image),
    )
    rejected = validate_chest_xray_suitability(
        image,
        "rejected-image",
        lambda _image: SuitabilityDecision(SuitabilityStatus.REJECTED, "Mock rejection."),
    )
    with pytest.raises(SuitabilityGateError):
        generate_gradcam(
            image,
            result,
            image_hash="rejected-image",
            validation=rejected,
        )


def test_streamlit_upload_runs_local_gate_and_offers_analysis_only_on_pass():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.file_uploader[0].set_value(("photo.png", png_bytes(), "image/png")).run(timeout=20)
    assert not app.exception
    assert any(
        "Image rejected:" in item.value
        or "Unable to verify" in item.value
        or "Image validation service unavailable" in item.value
        or "Chest X-ray input accepted" in item.value
        for item in [*app.error, *app.success, *app.warning]
    )

    app.radio[0].set_value("Analyze X-Ray").run()
    assert not app.exception
    accepted = any("Chest X-ray input accepted" in item.value for item in app.success)
    has_run_button = any("DenseNet-121 Analysis" in button.label for button in app.button)
    assert has_run_button is accepted
    if not accepted:
        assert not app.session_state.get("analysis")


def test_streamlit_rechecks_gate_before_inference_and_blocks_new_rejection(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from services import xray_gate_model

    decisions = iter(
        [
            SuitabilityDecision(SuitabilityStatus.ACCEPTED, "Mock upload pass."),
            SuitabilityDecision(SuitabilityStatus.REJECTED, "Mock recheck identifies Normal Photo."),
        ]
    )
    monkeypatch.setattr(xray_gate_model, "classify_chest_xray", lambda _image: next(decisions))

    def fail_if_model_loaded():
        pytest.fail("Pathology inference must not load after the fresh gate rejects the image")

    monkeypatch.setattr("services.inference.load_model", fail_if_model_loaded)

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.file_uploader[0].set_value(("photo.png", png_bytes(), "image/png")).run()
    app.radio[0].set_value("Analyze X-Ray").run()
    run_button = next(button for button in app.button if "DenseNet-121 Analysis" in button.label)
    run_button.click().run()

    assert not app.exception
    assert any("Image rejected:" in item.value for item in app.error)
    assert app.session_state.get("suitability_validation").status is SuitabilityStatus.REJECTED
    assert not app.session_state.get("analysis")


def test_streamlit_pdf_export_uses_original_image_and_gradcam_mapping(monkeypatch):
    from datetime import datetime, timezone

    from streamlit.testing.v1 import AppTest

    from services import xray_gate_model
    from services.inference import (
        PATHOLOGY_LABELS,
        AnalysisResult,
        Prediction,
        disease_model_revision,
    )
    from services.modality_validation import fingerprint_image

    monkeypatch.setattr(
        xray_gate_model,
        "classify_chest_xray",
        lambda _image: SuitabilityDecision(
            SuitabilityStatus.ACCEPTED,
            "Mock gate pass for PDF app integration test.",
        ),
    )

    def mock_inference(image, *, image_hash, validation):
        predictions = tuple(
            Prediction(label, 0.9 if index == 0 else 0.01, 0.5)
            for index, label in enumerate(PATHOLOGY_LABELS)
        )
        return AnalysisResult(
            model_name="Mock model (application integration test only)",
            checkpoint_id="mock-checkpoint",
            timestamp=datetime.now(timezone.utc).isoformat(),
            predictions=predictions,
            score_semantics="Mocked independent sigmoid score; not model output",
            raw_scores=tuple(0.0 for _ in PATHOLOGY_LABELS),
            pathologies=PATHOLOGY_LABELS,
            image_hash=image_hash,
            image_fingerprint=fingerprint_image(image),
            model_revision=disease_model_revision(),
        )

    monkeypatch.setattr("services.inference.run_inference", mock_inference)
    monkeypatch.setattr(
        "services.explainability.generate_gradcam",
        lambda image, *_args, **_kwargs: Image.new("RGB", image.size, color="red"),
    )

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20).run()
    app.file_uploader[0].set_value(("sample.png", png_bytes(), "image/png")).run(timeout=20)
    app.radio[0].set_value("Analyze X-Ray").run(timeout=20)
    run_button = next(button for button in app.button if "DenseNet-121 Analysis" in button.label)
    run_button.click().run(timeout=20)

    assert not app.exception
    assert app.session_state["analysis"].findings
    assert app.session_state["gradcams"]
    assert any("Download Research Dossier (PDF)" in button.label for button in app.download_button)
    assert not any("Could not compile PDF report dossier" in item.value for item in app.error)


def test_streamlit_corrupt_image_fails_closed_without_crashing():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    app.file_uploader[0].set_value(("broken.png", b"not a decodable image", "image/png")).run()

    assert not app.exception
    assert app.session_state.get("file_validation_error")
    assert not app.session_state.get("analysis")
