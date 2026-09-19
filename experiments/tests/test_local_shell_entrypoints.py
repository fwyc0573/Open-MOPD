"""Smoke tests for the supported local shell entry points.

These tests intentionally use ``/bin/echo`` instead of starting a trainer.  They
verify the user-facing contract (shell syntax, local-path validation, and command
construction) without requiring GPUs or model/data downloads.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = ("opd", "rl", "sft", "eval", "mt_opd")


def _run(name: str, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(ROOT / "scripts/local" / f"{name}.sh"), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_local_launchers_are_valid_shell() -> None:
    for name in LAUNCHERS:
        result = subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/local" / f"{name}.sh")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


def test_local_launchers_print_commands_without_running(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    teacher.mkdir()
    train.touch()
    val.touch()

    base = (
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--torchrun",
        "/bin/echo",
        "--dry-run",
    )
    for name in LAUNCHERS[:-1]:
        result = _run(name, *base)
        assert result.returncode == 0, (name, result.stderr)
        assert "[local]" in result.stdout
        assert "://" not in result.stdout
        assert "remote_submit" not in result.stdout

    result = _run(
        "mt_opd",
        *base,
        "--teacher",
        str(teacher),
        "--teacher",
        str(teacher),
        "--domains",
        "math,code",
    )
    assert result.returncode == 0, result.stderr
    assert "://" not in result.stdout
    assert "remote_submit" not in result.stdout


def test_run_mode_rejects_remote_uri(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    result = _run(
        "rl",
        "--model",
        "https://example.invalid/model",
        "--train",
        str(tmp_path / "train.parquet"),
        "--val",
        str(tmp_path / "val.parquet"),
        "--run",
    )
    assert result.returncode != 0
    assert "local filesystem path" in result.stderr


def test_run_mode_executes_with_local_paths_and_fake_binaries(tmp_path: Path) -> None:
    model = tmp_path / "model"
    teacher = tmp_path / "teacher"
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    model.mkdir()
    teacher.mkdir()
    train.touch()
    val.touch()

    base = (
        "--model",
        str(model),
        "--train",
        str(train),
        "--val",
        str(val),
        "--output",
        str(tmp_path / "output"),
        "--python",
        "/bin/echo",
        "--torchrun",
        "/bin/echo",
        "--run",
    )
    for name in LAUNCHERS[:-1]:
        result = _run(name, *base)
        assert result.returncode == 0, (name, result.stderr)
        assert "[local]" in result.stdout

    result = _run(
        "mt_opd",
        *base,
        "--teacher",
        str(teacher),
        "--teacher",
        str(teacher),
        "--domains",
        "math,code",
    )
    assert result.returncode == 0, result.stderr


def test_mt_run_mode_requires_explicit_external_ray_for_multiple_nodes(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    train = tmp_path / "train.parquet"
    val = tmp_path / "val.parquet"
    train.touch()
    val.touch()
    base = (
        "--run", "--python", "/bin/echo", "--torchrun", "/bin/echo",
        "--model", str(model), "--train", str(train), "--val", str(val),
        "--output", str(tmp_path / "output"), "--gpus", "8",
    )
    for nodes, address, allowed in (
        (1, None, True), (2, "127.0.0.1:6380", True),
        (2, None, False), (2, "auto", False), (2, "local", False),
        (2, "ray://127.0.0.1:10001", False),
    ):
        env = dict(os.environ)
        env.pop("RAY_ADDRESS", None)
        if address is not None:
            env["RAY_ADDRESS"] = address
        result = _run(
            "mt_opd", *base, "--nodes", str(nodes),
            "--teacher", str(model), "--teacher", str(model),
            "--domains", "math,code", "--", "trainer.total_training_steps=600", env=env,
        )
        assert (result.returncode == 0) is allowed, (nodes, address, result.stderr)
        if allowed:
            # The printed preview and the fake Python execution must both carry
            # the full configured horizon and requested native node count.
            assert result.stdout.count(f"trainer.nnodes={nodes}") == 2
            assert result.stdout.count("trainer.total_training_steps=600") == 2
        else:
            assert "NODES" in result.stderr
    for name in ("sft", "eval"):
        result = _run(name, *base, "--nodes", "2", env={**os.environ, "RAY_ADDRESS": "127.0.0.1:6380"})
        assert result.returncode != 0, name
        assert "NODES" in result.stderr
