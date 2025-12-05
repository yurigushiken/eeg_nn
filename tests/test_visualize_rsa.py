import numpy as np
import pandas as pd
from pathlib import Path


def _make_master_csv(tmp_path: Path) -> Path:
    data = [
        # OVERALL rows (two seeds)
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 70.0, "MacroF1": 68.0, "MinClassF1": 65.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 43, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 74.0, "MacroF1": 72.0, "MinClassF1": 69.0},
        {"ClassA": 11, "ClassB": 33, "Seed": 42, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 76.0, "MacroF1": 73.0, "MinClassF1": 71.0},
        {"ClassA": 22, "ClassB": 33, "Seed": 42, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 64.0, "MacroF1": 62.0, "MinClassF1": 59.0},
        {"ClassA": 22, "ClassB": 44, "Seed": 42, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 60.0, "MacroF1": 58.0, "MinClassF1": 55.0},
        {"ClassA": 33, "ClassB": 44, "Seed": 42, "Subject": "OVERALL", "RecordType": "overall", "Accuracy": 55.0, "MacroF1": 53.0, "MinClassF1": 50.0},
        # Subject rows (should be ignored during visualization)
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "Subject": "subject_01", "RecordType": "subject", "Accuracy": 68.0, "MacroF1": 66.0, "MinClassF1": 63.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "Subject": "subject_02", "RecordType": "subject", "Accuracy": 72.0, "MacroF1": 69.0, "MinClassF1": 66.0},
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "rsa_results_master.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_build_accuracy_matrix(tmp_path: Path):
    from scripts.visualize_rsa import build_accuracy_matrix

    csv_path = _make_master_csv(tmp_path)
    matrix, labels = build_accuracy_matrix(csv_path, metric="Accuracy", subject_filter="OVERALL")

    assert labels == [11, 22, 33, 44]
    # Expected average for 11v22 is (70 + 74)/2 = 72
    assert np.isclose(matrix[0, 1], 72.0)
    assert np.isclose(matrix[1, 0], 72.0)
    assert not np.isnan(matrix[0, 0])


def test_compute_mds_positions(tmp_path: Path):
    from scripts.visualize_rsa import build_accuracy_matrix, compute_mds_positions

    csv_path = _make_master_csv(tmp_path)
    matrix, labels = build_accuracy_matrix(csv_path, metric="Accuracy", subject_filter="OVERALL")

    positions = compute_mds_positions(matrix, labels)
    assert set(positions.columns) == {"label", "x", "y"}
    assert len(positions) == 4
    assert positions["label"].tolist() == labels


def test_plot_functions_create_files(tmp_path: Path):
    from scripts.visualize_rsa import (
        build_accuracy_matrix,
        compute_mds_positions,
        plot_rdm_heatmap,
        plot_mds_scatter,
    )

    csv_path = _make_master_csv(tmp_path)
    matrix, labels = build_accuracy_matrix(csv_path, metric="Accuracy", subject_filter="OVERALL")
    positions = compute_mds_positions(matrix, labels)

    heatmap_path = tmp_path / "heatmap.png"
    scatter_path = tmp_path / "mds.png"

    plot_rdm_heatmap(matrix, labels, heatmap_path)
    plot_mds_scatter(positions, scatter_path)

    assert heatmap_path.exists()
    assert scatter_path.exists()

