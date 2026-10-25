from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image

from config import CHECKPOINT_ID, MODEL_NAME
from services.modality_validation import (
    SuitabilityValidationResult,
    require_suitability_pass,
    fingerprint_image,
)
from services.preprocessing import preprocess_image

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when the pretrained model cannot be initialized."""


class InferenceError(RuntimeError):
    """Raised when a loaded model cannot produce a valid result."""


@dataclass(frozen=True)
class Prediction:
    label: str
    score: float


@dataclass(frozen=True)
class AnalysisResult:
    model_name: str
    checkpoint_id: str
    timestamp: str
    predictions: tuple[Prediction, ...]
    score_semantics: str
    raw_scores: tuple[float, ...]
    pathologies: tuple[str, ...]
    image_hash: str = ""
    image_fingerprint: str = ""


@lru_cache(maxsize=1)
def load_model():
    try:
        import torch
        import torchxrayvision as xrv

        model = xrv.models.DenseNet(weights=CHECKPOINT_ID)
        model.eval()
        if not getattr(model, "pathologies", None):
            raise ModelUnavailableError("The loaded model does not provide pathology labels.")
        return model, torch
    except ModelUnavailableError:
        raise
    except Exception as exc:
        logger.exception("Could not initialize the pretrained chest X-ray model.")
        raise ModelUnavailableError(
            "The pretrained model could not be loaded. Check the installation and network access, then try again."
        ) from exc


def run_inference(
    image: Image.Image,
    *,
    image_hash: str,
    validation: SuitabilityValidationResult | None,
) -> AnalysisResult:
    from datetime import datetime, timezone

    require_suitability_pass(validation, image_hash, image)
    try:
        model, torch = load_model()
        input_array = preprocess_image(image)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = model.to(device)
            tensor = torch.from_numpy(input_array).to(device)
            with torch.inference_mode():
                output = model(tensor)
        except Exception:
            if device.type != "cuda":
                raise
            logger.exception("GPU inference failed; retrying the same model on CPU.")
            device = torch.device("cpu")
            model = model.to(device)
            tensor = torch.from_numpy(input_array).to(device)
            with torch.inference_mode():
                output = model(tensor)
        values = output.detach().cpu().numpy()
        pathologies = tuple(str(label) for label in model.pathologies)
        if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] != len(pathologies):
            raise InferenceError("The model returned an unexpected output shape.")
        if not np.isfinite(values).all():
            raise InferenceError("The model returned non-finite scores.")

        raw_scores = tuple(float(value) for value in values[0])
        if getattr(model, "op_threshs", None) is not None:
            semantics = "Model output score (sigmoid and operating-point normalized; not a calibrated probability)"
        elif bool(getattr(model, "apply_sigmoid", False)):
            semantics = "Model output score (sigmoid-transformed; not a calibrated probability)"
        else:
            semantics = "Model output score (raw model output; not a calibrated probability)"
        predictions = tuple(
            Prediction(label=label, score=score)
            for label, score in zip(pathologies, raw_scores)
        )
        return AnalysisResult(
            model_name=MODEL_NAME,
            checkpoint_id=CHECKPOINT_ID,
            timestamp=datetime.now(timezone.utc).isoformat(),
            predictions=predictions,
            score_semantics=semantics,
            raw_scores=raw_scores,
            pathologies=pathologies,
            image_hash=image_hash,
            image_fingerprint=fingerprint_image(image),
        )
    except (ModelUnavailableError, InferenceError):
        raise
    except Exception as exc:
        logger.exception("Chest X-ray inference failed.")
        raise InferenceError(
            "Analysis could not be completed. Check the image and model installation, then try again."
        ) from exc
