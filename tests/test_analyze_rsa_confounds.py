import csv
from pathlib import Path
import numpy as np
import pandas as pd


def _write_stimuli_csv(path: Path) -> None:
    # Use only 'e' variants for the six numerosities 1-6
    rows = [
        {"filename": "1e.jpg", "dot_count": 1, "white_pixel_area": 100},
        {"filename": "2e.jpg", "dot_count": 2, "white_pixel_area": 150},
        {"filename": "3e.jpg", "dot_count": 3, "white_pixel_area": 200},
        {"filename": "4e.jpg", "dot_count": 4, "white_pixel_area": 250},
        {"filename": "5e.jpg", "dot_count": 5, "white_pixel_area": 300},
        {"filename": "6e.jpg", "dot_count": 6, "white_pixel_area": 400},
    ]
    fieldnames = ["filename", "dot_count", "white_pixel_area"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_master_csv(path: Path) -> None:
    # Build subject-level rows for all pairs; subject_01 tracks theory closely,
    # subject_02 is noisier.
    codes = [11, 22, 33, 44, 55, 66]
    data = []
    acc_map = {
        (11, 22): 80,
        (11, 33): 78,
        (11, 44): 75,
        (11, 55): 70,
        (11, 66): 65,
        (22, 33): 77,
        (22, 44): 75,
        (22, 55): 72,
        (22, 66): 70,
        (33, 44): 74,
        (33, 55): 72,
        (33, 66): 71,
        (44, 55): 70,
        (44, 66): 69,
        (55, 66): 68,
    }
    for (a, b), acc in acc_map.items():
        for subj in ("subj_01", "subj_02"):
            data.append(
                {
                    "ClassA": a,
                    "ClassB": b,
                    "Seed": 41,
                    "Subject": subj,
                    "RecordType": "subject",
                    "Accuracy": acc - (2 if subj == "subj_02" else 0),
                    "MacroF1": 0.0,
                    "MinClassF1": 0.0,
                    "n_trials": 10,
                    "n_correct": 7,
                    "ChanceRate": 50.0,
                }
            )
    pd.DataFrame(data).to_csv(path, index=False)


def test_confounds_pipeline_builds_rdms_and_stats(tmp_path: Path):
    stim_csv = tmp_path / "stimuli.csv"
    master_csv = tmp_path / "master.csv"
    _write_stimuli_csv(stim_csv)
    _write_master_csv(master_csv)

    from scripts.analyze_rsa_confounds import (
        build_theory_rdm,
        build_pixel_rdm,
        build_brain_rdms,
        partial_spearman,
        run_analysis,
    )

    codes = [11, 22, 33, 44, 55, 66]
    theory = build_theory_rdm(codes)
    pixel = build_pixel_rdm(stim_csv, codes)
    brain_rdms = build_brain_rdms(master_csv, codes)

    assert theory.shape == (6, 6)
    assert pixel.shape == (6, 6)
    assert len(brain_rdms) == 2  # two subjects

    subj_vec = next(iter(brain_rdms.values()))
    theory_vec = theory[np.tril_indices(len(codes), k=-1)]
    pixel_vec = pixel[np.tril_indices(len(codes), k=-1)]
    r_raw, r_partial = partial_spearman(subj_vec, theory_vec, pixel_vec)
    assert not np.isnan(r_raw)
    assert not np.isnan(r_partial)

    out_dir = tmp_path / "confounds"
    stats_df = run_analysis(master_csv, stim_csv, out_dir, codes=codes, baseline=0.4)
    assert {"Subject", "Raw_Spearman_R", "Partial_Spearman_R"}.issubset(stats_df.columns)
    summary = stats_df[stats_df["Subject"] == "SUMMARY"]
    assert not summary.empty

