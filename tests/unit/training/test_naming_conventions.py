from __future__ import annotations

import pytest


@pytest.mark.unit
def test_training_runner_enforces_naming_conventions():
    from code import training_runner

    format_subject = getattr(training_runner, "format_subject_id", None)
    format_fold = getattr(training_runner, "format_fold_id", None)
    validate_artifact_name = getattr(training_runner, "validate_artifact_name", None)

    if not callable(format_subject) or not callable(format_fold) or not callable(validate_artifact_name):
        pytest.fail(
            "training_runner must expose format_subject_id, format_fold_id, and validate_artifact_name helpers"
        )

    assert format_subject(7) == "subject_07"
    assert format_subject(23) == "subject_23"
    assert format_fold(3) == "fold_03"
    assert format_fold(12) == "fold_12"

    good_names = [
        "metrics_outer.csv",
        "stats/glmm_results.json",
        "predictions/outer_predictions.csv",
        "logs/runtime.jsonl",
    ]
    for name in good_names:
        validate_artifact_name(name)

    with pytest.raises(ValueError):
        validate_artifact_name("metricsOuter.csv")

    with pytest.raises(ValueError):
        validate_artifact_name("fold-1.txt")

