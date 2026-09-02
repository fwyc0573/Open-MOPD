"""Regression test for entropy computation on non-contiguous logits."""

import ast
from pathlib import Path

import pytest
import torch


def _load_entropy_method():
    source_path = Path(__file__).parents[2] / "training/verl/verl/workers/fsdp_workers.py"
    tree = ast.parse(source_path.read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_entropy_safe"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace[method.name]


def test_entropy_accepts_non_contiguous_logits():
    logits = torch.randn(2, 4, 7).transpose(0, 1)
    assert not logits.is_contiguous()

    entropy = _load_entropy_method()(object(), logits, chunk_size=3)

    assert entropy.shape == (4, 2)
    assert torch.isfinite(entropy).all()
