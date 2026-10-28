from datetime import datetime, timezone
from io import BytesIO

from PIL import Image
from pypdf import PdfReader
import pytest

from services.inference import AnalysisResult, Prediction
from services.modality_validation import (
    SuitabilityDecision,
    SuitabilityGateError,
    SuitabilityStatus,
    fingerprint_image,
    validate_chest_xray_suitability,
)
from services.report_generator import generate_pdf_report

TEST_IMAGE_HASH = "test-image-sha256"


def test_pdf_report_is_a_valid_pdf_with_mocked_unit_test_outputs():
    image = Image.new("L", (48, 32), color=120)
    result = AnalysisResult(
        model_name="Mock model (unit test only)",
        checkpoint_id="test-checkpoint",
        timestamp=datetime.now(timezone.utc).isoformat(),
        predictions=(Prediction("Test label", 0.42, 0.3),),
        score_semantics="Mocked independent sigmoid score; not a model prediction",
        raw_scores=(0.42,),
        pathologies=("Test label",),
        image_hash=TEST_IMAGE_HASH,
        image_fingerprint=fingerprint_image(image),
    )
    validation = validate_chest_xray_suitability(
        image,
        TEST_IMAGE_HASH,
        lambda _image: SuitabilityDecision(
            SuitabilityStatus.ACCEPTED,
            "Mock accepted for report unit test only.",
            xray_score=0.82,
            classifier_id="mock-classifier-unit-test",
        ),
    )
    pdf = generate_pdf_report(
        result=result,
        filename="test.png",
        original_image=image,
        gradcam_images={"Test label": image},
        image_hash=TEST_IMAGE_HASH,
        validation=validation,
    )
    assert pdf.startswith(b"%PDF-")
    assert pdf.rstrip().endswith(b"%%EOF")
    assert len(pdf) > 500
    reader = PdfReader(BytesIO(pdf))
    assert len(reader.pages) == 1
    extracted_text = reader.pages[0].extract_text()
    assert "test-checkpoint" in extracted_text
    assert "mock-classifier-unit-test" in extracted_text
    assert "0.8200 (not calibrated)" in extracted_text
    assert "Test label" in extracted_text
    assert "0.3000" in extracted_text
    assert "Meets cutoff" in extracted_text


def test_pdf_includes_each_threshold_finding_description_and_marks_missing_attribution():
    image = Image.new("RGB", (48, 32), color="gray")
    result = AnalysisResult(
        model_name="Mock model (unit test only)",
        checkpoint_id="test-checkpoint",
        timestamp=datetime.now(timezone.utc).isoformat(),
        predictions=(
            Prediction("Effusion", 0.72, 0.25),
            Prediction("Pneumonia", 0.63, 0.05),
            Prediction("Edema", 0.02, 0.05),
        ),
        score_semantics="Mocked independent sigmoid score; not a model prediction",
        raw_scores=(0.0, 0.0, 0.0),
        pathologies=("Effusion", "Pneumonia", "Edema"),
        image_hash=TEST_IMAGE_HASH,
        image_fingerprint=fingerprint_image(image),
    )
    validation = validate_chest_xray_suitability(
        image,
        TEST_IMAGE_HASH,
        lambda _image: SuitabilityDecision(
            SuitabilityStatus.ACCEPTED,
            "Mock accepted for report unit test only.",
        ),
    )

    pdf = generate_pdf_report(
        result=result,
        filename="test.png",
        original_image=image,
        gradcam_images={"Effusion": image},
        image_hash=TEST_IMAGE_HASH,
        validation=validation,
    )
    extracted_text = "\n".join(page.extract_text() for page in PdfReader(BytesIO(pdf)).pages)

    assert extracted_text.index("Effusion") < extracted_text.index("Pneumonia")
    assert "blunting of a costophrenic angle" in extracted_text
    assert "air-space opacity or consolidation" in extracted_text
    assert "Grad-CAM unavailable" in extracted_text
    assert "No usable positive attribution map was" in extracted_text
    assert "produced for this finding" in extracted_text


def test_pdf_export_is_blocked_for_rejected_or_mismatched_image():
    image = Image.new("L", (48, 32), color=120)
    result = AnalysisResult(
        model_name="Mock model (unit test only)",
        checkpoint_id="test-checkpoint",
        timestamp=datetime.now(timezone.utc).isoformat(),
        predictions=(Prediction("Test label", 0.42, 0.3),),
        score_semantics="Mock test score",
        raw_scores=(0.42,),
        pathologies=("Test label",),
        image_hash=TEST_IMAGE_HASH,
        image_fingerprint=fingerprint_image(image),
    )
    rejected = validate_chest_xray_suitability(
        image,
        TEST_IMAGE_HASH,
        lambda _image: SuitabilityDecision(SuitabilityStatus.REJECTED, "Mock rejection."),
    )
    with pytest.raises(SuitabilityGateError):
        generate_pdf_report(
            result=result,
            filename="test.png",
            original_image=image,
            gradcam_images={"Test label": image},
            image_hash=TEST_IMAGE_HASH,
            validation=rejected,
        )

    accepted = validate_chest_xray_suitability(
        image,
        "different-image-hash",
        lambda _image: SuitabilityDecision(
            SuitabilityStatus.ACCEPTED,
            "Mock acceptance for another image.",
        ),
    )
    with pytest.raises(SuitabilityGateError):
        generate_pdf_report(
            result=result,
            filename="test.png",
            original_image=image,
            gradcam_images={"Test label": image},
            image_hash=TEST_IMAGE_HASH,
            validation=accepted,
        )


def test_pdf_generator_rejects_positional_image_arguments():
    with pytest.raises(TypeError):
        generate_pdf_report(None, "test.png", Image.new("RGB", (8, 8)), {})
