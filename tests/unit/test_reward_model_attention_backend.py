"""Regression tests for the RewardModelWorker attention backend selection."""

import ast
from pathlib import Path
from types import SimpleNamespace


def _load_attention_selector():
    source_path = Path(__file__).parents[2] / "training/verl/verl/workers/fsdp_workers.py"
    tree = ast.parse(source_path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_reward_model_attention_implementation"
    )
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace[function.name]


def test_reward_model_attention_backend_honors_explicit_override():
    config = SimpleNamespace(model={"override_config": {"attn_implementation": "eager"}})

    assert _load_attention_selector()(config) == "eager"


def test_reward_model_attention_backend_keeps_flash_default():
    config = SimpleNamespace(model={"override_config": {}})

    assert _load_attention_selector()(config) == "flash_attention_2"
