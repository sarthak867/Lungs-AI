from __future__ import annotations

import hashlib
from typing import MutableMapping

from services.modality_validation import SuitabilityStatus, SuitabilityValidationResult
from services.preprocessing import ValidatedImage
from services.xray_gate_model import classifier_revision

ANALYSIS_STATE_KEYS = (
    "suitability_validation",
    "suitability_validation_hash",
    "suitability_validation_version",
    "analysis",
    "analysis_hash",
    "gradcam",
    "gradcams",
    "gradcam_errors",
    "gradcam_label",
    "gradcam_error",
    "report_bytes",
    "report_hash",
)
RESULT_STATE_KEYS = (
    "analysis",
    "analysis_hash",
    "gradcam",
    "gradcams",
    "gradcam_errors",
    "gradcam_label",
    "gradcam_error",
    "report_bytes",
    "report_hash",
)
UPLOAD_STATE_KEYS = (
    "image_bytes",
    "image_hash",
    "image_name",
    "image_width",
    "image_height",
    "file_validation_error",
)


def content_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clear_analysis_state(state: MutableMapping) -> None:
    for key in ANALYSIS_STATE_KEYS:
        state.pop(key, None)


def clear_result_state(state: MutableMapping) -> None:
    for key in RESULT_STATE_KEYS:
        state.pop(key, None)


def clear_image_state(state: MutableMapping) -> None:
    clear_analysis_state(state)
    for key in UPLOAD_STATE_KEYS:
        state.pop(key, None)


def register_valid_upload(
    state: MutableMapping,
    data: bytes,
    filename: str,
    validated_image: ValidatedImage,
) -> str:
    digest = content_sha256(data)
    if (
        digest != state.get("image_hash")
        or filename != state.get("image_name")
    ):
        clear_analysis_state(state)
    state["image_bytes"] = data
    state["image_hash"] = digest
    state["image_name"] = filename
    state["image_width"] = validated_image.width
    state["image_height"] = validated_image.height
    state.pop("file_validation_error", None)
    return digest


def register_invalid_upload(state: MutableMapping, reason: str) -> None:
    clear_image_state(state)
    state["file_validation_error"] = reason


def attach_suitability_result(
    state: MutableMapping,
    result: SuitabilityValidationResult,
) -> bool:
    if result.content_hash != state.get("image_hash"):
        clear_analysis_state(state)
        return False
    if result.status is not SuitabilityStatus.ACCEPTED:
        clear_result_state(state)
    state["suitability_validation"] = result
    state["suitability_validation_hash"] = result.content_hash
    state["suitability_validation_version"] = classifier_revision()
    return True
