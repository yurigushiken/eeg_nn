from __future__ import annotations

import pytest


@pytest.mark.integration
def test_data_finalization_pipeline(materialized_dataset, tmp_path):
    from code import data_finalization

    finalize = getattr(data_finalization, "finalize_materialized_dataset", None)
    if finalize is None:
        pytest.fail("finalize_materialized_dataset must be implemented to finalize materialized datasets")

    result = finalize(
        source_dir=materialized_dataset.materialized_dir,
        output_root=tmp_path / "finalized",
        min_trials_per_class=5,
        channel_strategy="intersection",
        metadata_only=False,
    )

    for attr in ("codebook_path", "exclusion_report", "kept_subjects", "excluded_subjects"):
        assert hasattr(result, attr), f"Result should expose {attr}"

    assert result.codebook_path.exists(), "Global condition codebook must be persisted"
    assert result.exclusion_report.exists(), "Exclusion report should be written"
    kept_subjects = [spec.subject_id for spec in materialized_dataset.specs if all(v >= 5 for v in spec.class_counts.values())]
    for subject_id in kept_subjects:
        expected = result.output_root / f"sub-{subject_id:02d}_finalized-epo.fif"
        assert expected.exists(), "Finalized .fif should be materialized for each retained subject"
    assert 3 in result.excluded_subjects, "Subjects below trial threshold must be reported"
