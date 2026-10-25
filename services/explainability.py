from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from services.inference import AnalysisResult, ModelUnavailableError, load_model
from services.modality_validation import (
    SuitabilityValidationResult,
    require_suitability_pass,
)
from services.preprocessing import center_crop_image, preprocess_image

logger = logging.getLogger(__name__)


class ExplainabilityUnavailableError(RuntimeError):
    """Raised when a trustworthy Grad-CAM cannot be produced."""


def generate_gradcam(
    image: Image.Image,
    result: AnalysisResult,
    target_index: int = 0,
    *,
    image_hash: str,
    validation: SuitabilityValidationResult | None,
) -> Image.Image:
    require_suitability_pass(validation, image_hash, image)
    if result.image_hash != image_hash or result.image_fingerprint != validation.image_fingerprint:
        raise ExplainabilityUnavailableError("Grad-CAM requires analysis results for this exact image.")
    activation = None
    handle = None
    try:
        model, torch = load_model()
        features = getattr(model, "features", None)
        target_layer = getattr(features, "denseblock4", None)
        if target_layer is None:
            raise ExplainabilityUnavailableError(
                "Grad-CAM is unavailable because the loaded model has no supported target layer."
            )

        def capture_activation(_module, _inputs, output):
            nonlocal activation
            activation = output
            output.retain_grad()

        handle = target_layer.register_forward_hook(capture_activation)
        model.zero_grad(set_to_none=True)
        tensor = torch.from_numpy(preprocess_image(image)).to(next(model.parameters()).device)
        output = model(tensor)
        if (
            output.ndim != 2
            or output.shape[0] != 1
            or not 0 <= target_index < output.shape[1]
            or target_index >= len(result.pathologies)
        ):
            raise ExplainabilityUnavailableError("Grad-CAM could not select a valid model output.")
        output[0, target_index].backward()
        if activation is None or activation.grad is None:
            raise ExplainabilityUnavailableError("Grad-CAM gradients were not available for this model.")

        gradient = activation.grad.detach().mean(dim=(2, 3), keepdim=True)
        activation_values = activation.detach()
        cam = torch.relu((gradient * activation_values).sum(dim=1, keepdim=True))
        cropped_image = center_crop_image(image)
        cam = torch.nn.functional.interpolate(
            cam, size=cropped_image.size[::-1], mode="bilinear", align_corners=False
        )[0, 0]
        heatmap = cam.cpu().numpy()
        maximum = float(heatmap.max())
        if not np.isfinite(heatmap).all() or maximum <= 0:
            raise ExplainabilityUnavailableError("Grad-CAM produced no usable attribution map.")
        heatmap = np.clip(heatmap / maximum, 0.0, 1.0)

        import matplotlib

        colormap = matplotlib.colormaps["magma"]
        rgb = (colormap(heatmap)[..., :3] * 255).astype(np.uint8)
        color = Image.fromarray(rgb, mode="RGB")
        base = image.convert("L").convert("RGB")
        cropped_base = cropped_image.convert("RGB")
        overlay_crop = Image.blend(cropped_base, color, alpha=0.42)
        left = (image.width - cropped_image.width) // 2
        top = (image.height - cropped_image.height) // 2
        base.paste(overlay_crop, (left, top))
        return base
    except ExplainabilityUnavailableError:
        raise
    except ModelUnavailableError as exc:
        raise ExplainabilityUnavailableError(
            "Grad-CAM is unavailable because the pretrained model could not be loaded."
        ) from exc
    except Exception as exc:
        logger.exception("Grad-CAM generation failed.")
        raise ExplainabilityUnavailableError(
            "A valid Grad-CAM could not be generated for this analysis."
        ) from exc
    finally:
        if handle is not None:
            handle.remove()
