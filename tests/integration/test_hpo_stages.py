from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.integration
def test_stage_handoff_requires_prior_evidence(tmp_path):
    from code import training_runner

    validator = getattr(training_runner, "validate_stage_handoff", None)
    if validator is None:
        pytest.fail("validate_stage_handoff must be implemented to coordinate sequential Optuna stages")

    with pytest.raises(ValueError):
        validator(stage="stage_2", previous_stage_dir=None)

    stage1_dir = tmp_path / "stage_1_champion"
    evidence1 = stage1_dir / "evidence"
    evidence1.mkdir(parents=True, exist_ok=True)
    champion_cfg = stage1_dir / "resolved_config.yaml"
    champion_cfg.write_text("seed: 1337\n")
    (evidence1 / "outer_eval_metrics.csv").write_text("fold,acc\n1,0.75\n")

    handoff = validator(stage="stage_2", previous_stage_dir=stage1_dir)
    assert "champion_config" in handoff
    assert handoff["champion_config"] == champion_cfg
    assert handoff["evidence_dir"] == evidence1

    stage2_dir = tmp_path / "stage_2_champion"
    evidence2 = stage2_dir / "evidence"
    evidence2.mkdir(parents=True, exist_ok=True)
    (stage2_dir / "resolved_config.yaml").write_text("seed: 2025\n")
    (evidence2 / "outer_eval_metrics.csv").write_text("fold,acc\n1,0.80\n")

    handoff2 = validator(stage="stage_3", previous_stage_dir=stage2_dir)
    assert handoff2["champion_config"].exists()
    assert handoff2["evidence_dir"].exists()

    with pytest.raises(ValueError):
        validator(stage="final_loso", previous_stage_dir=None)

