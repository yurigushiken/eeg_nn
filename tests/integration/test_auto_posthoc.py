from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT / "train.py"


def _load_train_module():
    if not TRAIN_PATH.exists():
        pytest.fail(f"train.py missing at {TRAIN_PATH}")
    spec = importlib.util.spec_from_file_location("train_module", TRAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.integration
def test_train_triggers_posthoc_when_enabled(monkeypatch, tmp_path):
    train_mod = _load_train_module()
    orchestrate = getattr(train_mod, "run_pipeline_with_posthoc", None)
    if orchestrate is None:
        pytest.fail("train.run_pipeline_with_posthoc must orchestrate training and post-hoc analysis")

    called = {"train": 0, "posthoc": 0}

    def fake_execute(config, output_root):
        called["train"] += 1
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": run_dir, "config": config}

    def fake_posthoc(run_dir, config):
        called["posthoc"] += 1
        assert Path(run_dir).exists(), "Post-hoc should receive existing run directory"

    monkeypatch.setattr(train_mod, "execute_training_run", fake_execute, raising=False)
    monkeypatch.setattr(train_mod, "trigger_posthoc_analysis", fake_posthoc, raising=False)

    config = {"stats": {"run_posthoc_after_train": True}}
    result = orchestrate(config, output_root=tmp_path)

    assert called["train"] == 1, "Training phase must execute exactly once"
    assert called["posthoc"] == 1, "Post-hoc analysis must trigger when enabled"
    assert result.get("posthoc_triggered") is True


@pytest.mark.integration
def test_posthoc_bypass_when_disabled(monkeypatch, tmp_path):
    train_mod = _load_train_module()
    orchestrate = getattr(train_mod, "run_pipeline_with_posthoc", None)
    if orchestrate is None:
        pytest.fail("train.run_pipeline_with_posthoc must orchestrate training and post-hoc analysis")

    called = {"train": 0, "posthoc": 0}

    def fake_execute(config, output_root):
        called["train"] += 1
        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": run_dir, "config": config}

    def fake_posthoc(*args, **kwargs):
        called["posthoc"] += 1

    monkeypatch.setattr(train_mod, "execute_training_run", fake_execute, raising=False)
    monkeypatch.setattr(train_mod, "trigger_posthoc_analysis", fake_posthoc, raising=False)

    config = {"stats": {"run_posthoc_after_train": False}}
    result = orchestrate(config, output_root=tmp_path)

    assert called["train"] == 1
    assert called["posthoc"] == 0, "Post-hoc analysis must not run when disabled"
    assert result.get("posthoc_triggered") is False

