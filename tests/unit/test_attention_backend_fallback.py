"""Regression tests for CUDA attention helpers without flash-attn installed."""

import importlib
import sys


def test_padding_module_falls_back_when_cuda_has_no_flash_attn(monkeypatch):
    import verl.utils.device as device
    import verl.utils.npu_utils as fallback

    monkeypatch.setattr(device, "is_cuda_available", True)
    sys.modules.pop("verl.workers.roles.utils.padding", None)
    padding = importlib.import_module("verl.workers.roles.utils.padding")

    assert padding.pad_input is fallback.pad_input
    assert padding.unpad_input is fallback.unpad_input


def test_attention_utils_falls_back_when_cuda_has_no_flash_attn(monkeypatch):
    import verl.utils.attention_utils as attention_utils
    import verl.utils.device as device
    import verl.utils.npu_utils as fallback

    monkeypatch.setattr(device, "is_cuda_available", True)
    attention_utils._index_first_axis = None
    attention_utils._pad_input = None
    attention_utils._rearrange = None
    attention_utils._unpad_input = None

    assert attention_utils.pad_input.__module__ == attention_utils.__name__
    assert attention_utils.pad_input(
        torch_ones := __import__("torch").ones(1, 1, 1),
        indices=__import__("torch").tensor([0]),
        batch=1,
        seqlen=1,
    ).shape == fallback.pad_input(torch_ones, __import__("torch").tensor([0]), 1, 1).shape
