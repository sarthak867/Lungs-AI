import pytest
import sys
import types

import numpy as np
from PIL import Image

from services.inference import ModelUnavailableError, load_model, run_inference
from services.modality_validation import (
    SuitabilityDecision,
    SuitabilityGateError,
    SuitabilityStatus,
    validate_chest_xray_suitability,
)


def test_missing_model_dependency_is_reported_as_unavailable(monkeypatch):
    load_model.cache_clear()
    import builtins

    original_import = builtins.__import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("mocked missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)
    with pytest.raises(ModelUnavailableError):
        load_model()
    load_model.cache_clear()


def test_unavailable_checkpoint_weights_are_reported(monkeypatch):
    torch_stub = types.ModuleType("torch")
    xrv_stub = types.ModuleType("torchxrayvision")

    def fail_to_load_weights(**_kwargs):
        raise OSError("mocked checkpoint download failure")

    xrv_stub.models = types.SimpleNamespace(DenseNet=fail_to_load_weights)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torchxrayvision", xrv_stub)
    load_model.cache_clear()
    with pytest.raises(ModelUnavailableError, match="pretrained model could not be loaded"):
        load_model()
    load_model.cache_clear()


@pytest.mark.parametrize(
    "status",
    [
        SuitabilityStatus.REJECTED,
        SuitabilityStatus.UNABLE_TO_VERIFY,
        SuitabilityStatus.SERVICE_UNAVAILABLE,
    ],
)
def test_pathology_inference_is_never_called_without_suitability_pass(monkeypatch, status):
    def fail_if_called():
        pytest.fail("pathology model must not load for rejected input")

    monkeypatch.setattr("services.inference.load_model", fail_if_called)
    image = Image.new("RGB", (16, 16))
    rejected = validate_chest_xray_suitability(
        image,
        "image-hash",
        lambda _image: SuitabilityDecision(status, "mock non-pass"),
    )
    with pytest.raises(SuitabilityGateError):
        run_inference(
            image,
            image_hash="image-hash",
            validation=rejected,
        )


def test_pathology_model_is_reached_only_after_matching_mock_gate_pass(monkeypatch):
    calls = []

    class MockTensor:
        def to(self, _device):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray([[0.25]], dtype=np.float32)

    class MockModel:
        pathologies = ["Mock label"]
        op_threshs = None
        apply_sigmoid = False

        def to(self, _device):
            return self

        def __call__(self, _tensor):
            calls.append("forward")
            return MockTensor()

    class MockTorch:
        class device:
            def __init__(self, _name):
                self.type = "cpu"

        @staticmethod
        def from_numpy(_array):
            return MockTensor()

        @staticmethod
        def inference_mode():
            from contextlib import nullcontext

            return nullcontext()

    MockTorch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr("services.inference.load_model", lambda: (MockModel(), MockTorch))
    image = Image.new("L", (16, 16))
    accepted = validate_chest_xray_suitability(
        image,
        "image-hash",
        lambda _image: SuitabilityDecision(
            SuitabilityStatus.ACCEPTED,
            "mock accepted for application logic test",
        ),
    )
    result = run_inference(
        image,
        image_hash="image-hash",
        validation=accepted,
    )
    assert calls == ["forward"]
    assert result.image_hash == "image-hash"
