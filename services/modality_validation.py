from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from PIL import Image

logger = logging.getLogger(__name__)


class SuitabilityStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNABLE_TO_VERIFY = "unable_to_verify"
    SERVICE_UNAVAILABLE = "service_unavailable"


@dataclass(frozen=True)
class SuitabilityDecision:
    status: SuitabilityStatus
    reason: str = ""
    xray_score: float | None = None
    classifier_id: str | None = None


@dataclass(frozen=True)
class SuitabilityValidationResult:
    status: SuitabilityStatus
    content_hash: str
    reason: str
    classifier_id: str | None = None
    image_fingerprint: str = ""
    xray_score: float | None = None

    @property
    def accepted(self) -> bool:
        return self.status is SuitabilityStatus.ACCEPTED


class SuitabilityServiceUnavailable(RuntimeError):
    """Raised when no configured radiograph-suitability model can be used."""


class SuitabilityGateError(RuntimeError):
    """Raised when downstream analysis is attempted without a matching gate pass."""


def validate_chest_xray_suitability(
    image: Image.Image,
    content_hash: str,
    classifier: Callable[[Image.Image], SuitabilityDecision] | None = None,
) -> SuitabilityValidationResult:
    """Run a configured suitability classifier; never infer modality from image properties."""
    if classifier is None:
        try:
            from services.xray_gate_model import classify_chest_xray

            classifier = classify_chest_xray
        except ImportError as exc:
            logger.exception("Could not import the configured local chest X-ray classifier.")
            return SuitabilityValidationResult(
                status=SuitabilityStatus.SERVICE_UNAVAILABLE,
                content_hash=content_hash,
                reason="The chest X-ray validation model could not be initialized.",
                image_fingerprint=fingerprint_image(image),
            )
    fingerprint = fingerprint_image(image)
    try:
        decision = classifier(image)
        if not isinstance(decision, SuitabilityDecision):
            raise TypeError("Classifier must return a SuitabilityDecision.")
        if decision.status is SuitabilityStatus.SERVICE_UNAVAILABLE:
            return SuitabilityValidationResult(
                status=decision.status,
                content_hash=content_hash,
                reason=decision.reason or "The image validation service is unavailable.",
                image_fingerprint=fingerprint,
                xray_score=decision.xray_score,
            )
        return SuitabilityValidationResult(
            status=decision.status,
            content_hash=content_hash,
            reason=decision.reason,
            classifier_id=decision.classifier_id or getattr(classifier, "classifier_id", None),
            image_fingerprint=fingerprint,
            xray_score=decision.xray_score,
        )
    except SuitabilityServiceUnavailable as exc:
        return SuitabilityValidationResult(
            status=SuitabilityStatus.SERVICE_UNAVAILABLE,
            content_hash=content_hash,
            reason=str(exc) or "The image validation service is unavailable.",
            image_fingerprint=fingerprint,
        )
    except Exception as exc:
        logger.exception("Chest X-ray suitability validation failed.")
        return SuitabilityValidationResult(
            status=SuitabilityStatus.UNABLE_TO_VERIFY,
            content_hash=content_hash,
            reason="The image type could not be verified.",
            image_fingerprint=fingerprint,
        )


def fingerprint_image(image: Image.Image) -> str:
    normalized = image.convert("L")
    digest = hashlib.sha256()
    digest.update(f"{normalized.width}x{normalized.height}:L:".encode("ascii"))
    digest.update(normalized.tobytes())
    return digest.hexdigest()


def require_suitability_pass(
    validation: SuitabilityValidationResult | None,
    content_hash: str,
    image: Image.Image,
) -> None:
    if (
        validation is None
        or validation.status is not SuitabilityStatus.ACCEPTED
        or validation.content_hash != content_hash
        or not validation.image_fingerprint
        or validation.image_fingerprint != fingerprint_image(image)
    ):
        raise SuitabilityGateError("A matching accepted chest X-ray suitability result for this image is required.")


def suitability_message(result: SuitabilityValidationResult) -> str:
    if result.status is SuitabilityStatus.ACCEPTED:
        return "Chest X-ray input accepted for research analysis. This is not a guarantee of authenticity or clinical usability."
    if result.status is SuitabilityStatus.REJECTED:
        return "Image rejected: this does not appear to be a supported chest X-ray. Upload a frontal chest X-ray in PNG or JPEG format."
    if result.status is SuitabilityStatus.SERVICE_UNAVAILABLE:
        return "Image validation service unavailable. Pathology analysis is blocked; verify xray_classifier.pth and the PyTorch installation."
    return "Unable to verify image type; analysis blocked. Upload a frontal chest X-ray in PNG or JPEG format."
