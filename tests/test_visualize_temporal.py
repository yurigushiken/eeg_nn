"""
Test suite for temporal RSA visualization.

Validates the temporal emergence plotting and ensures publication-ready output.
"""
import pandas as pd
import tempfile
from pathlib import Path

import pytest


def test_load_temporal_results():
    """Test loading and parsing temporal results CSV."""
    from scripts.visualize_temporal import load_temporal_results

    # Create mock CSV
    csv_content = """ClassA,ClassB,Seed,TimeWindow_Start,TimeWindow_End,TimeWindow_Center,Accuracy,MacroF1,MinClassF1
11,22,42,0,50,25,49.5,48.5,45.0
11,22,42,20,70,45,55.0,54.0,52.0
11,22,43,0,50,25,51.0,50.0,48.0
11,22,43,20,70,45,56.0,55.0,53.0
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = Path(f.name)

    try:
        df = load_temporal_results(csv_path)

        # Check columns
        assert "TimeWindow_Center" in df.columns
        assert "Accuracy" in df.columns
        assert "ClassA" in df.columns
        assert "Seed" in df.columns

        # Check data types (accept numpy int64/float64)
        assert df["TimeWindow_Center"].dtype.kind in ['i', 'f']  # int or float kind
        assert df["Accuracy"].dtype.kind == 'f'  # float kind

        # Check shape
        assert len(df) == 4
    finally:
        csv_path.unlink()


def test_aggregate_temporal_data():
    """Test aggregating temporal data across seeds."""
    from scripts.visualize_temporal import aggregate_temporal_data

    df = pd.DataFrame({
        "ClassA": [11, 11, 11, 11],
        "ClassB": [22, 22, 22, 22],
        "Seed": [42, 42, 43, 43],
        "TimeWindow_Center": [25, 45, 25, 45],
        "Accuracy": [49.5, 55.0, 51.0, 56.0],
        "MacroF1": [48.0, 54.0, 49.0, 55.0],
        "MinClassF1": [45.0, 52.0, 46.0, 53.0],
    })

    agg = aggregate_temporal_data(df, pair=(11, 22))

    # Should have 2 time windows
    assert len(agg) == 2

    # Check aggregation
    assert agg.loc[agg["TimeWindow_Center"] == 25, "Mean_Accuracy"].values[0] == 50.25
    assert agg.loc[agg["TimeWindow_Center"] == 45, "Mean_Accuracy"].values[0] == 55.5

    # Check SEM calculation
    assert "SEM_Accuracy" in agg.columns


def test_format_pair_label():
    """Test formatting pair labels for display."""
    from scripts.visualize_temporal import format_pair_label

    assert format_pair_label(11, 22) == "1 vs 2"
    assert format_pair_label(44, 55) == "4 vs 5"
    assert format_pair_label(11, 66) == "1 vs 6"


def test_temporal_plot_structure():
    """Test that temporal plot has correct structure."""
    from scripts.visualize_temporal import create_temporal_plot
    import matplotlib.pyplot as plt

    df = pd.DataFrame({
        "TimeWindow_Center": [25, 45, 65, 85, 105, 125],
        "Mean_Accuracy": [50.0, 52.0, 55.0, 58.0, 60.0, 57.0],
        "SEM_Accuracy": [2.0, 2.5, 2.0, 1.5, 1.8, 2.2],
    })

    fig, ax = create_temporal_plot(df, pair_label="1 vs 2")

    # Check plot properties
    assert ax.get_xlabel() == "Time (ms)"
    assert "Accuracy" in ax.get_ylabel()
    assert ax.get_title() == "Temporal Emergence: 1 vs 2"

    # Check chance line exists
    lines = ax.get_lines()
    assert len(lines) >= 2  # Data line + chance line

    plt.close(fig)


def test_find_peak_latency():
    """Test finding peak accuracy latency."""
    from scripts.visualize_temporal import find_peak_latency

    df = pd.DataFrame({
        "TimeWindow_Center": [25, 45, 65, 85, 105],
        "Mean_Accuracy": [50.0, 52.0, 58.0, 55.0, 53.0],
    })

    peak_time, peak_acc = find_peak_latency(df)

    assert peak_time == 65
    assert peak_acc == 58.0


def test_temporal_emergence_onset():
    """Test detecting emergence onset (first above-chance time)."""
    from scripts.visualize_temporal import find_emergence_onset

    df = pd.DataFrame({
        "TimeWindow_Center": [25, 45, 65, 85, 105],
        "Mean_Accuracy": [49.0, 51.0, 52.5, 55.0, 57.0],
    })

    # Onset should be first point >= threshold
    onset = find_emergence_onset(df, threshold=51.0)
    assert onset == 45  # First point at 51.0% (exactly at threshold)


def test_plot_multiple_pairs():
    """Test plotting multiple pairs on same axes."""
    from scripts.visualize_temporal import create_multi_pair_plot
    import matplotlib.pyplot as plt

    df1 = pd.DataFrame({
        "TimeWindow_Center": [25, 45, 65],
        "Mean_Accuracy": [50.0, 55.0, 60.0],
        "SEM_Accuracy": [2.0, 2.5, 2.0],
    })

    df2 = pd.DataFrame({
        "TimeWindow_Center": [25, 45, 65],
        "Mean_Accuracy": [51.0, 53.0, 56.0],
        "SEM_Accuracy": [1.8, 2.2, 1.9],
    })

    pairs_data = {
        "1 vs 2": df1,
        "4 vs 5": df2,
    }

    fig, ax = create_multi_pair_plot(pairs_data)

    # Should have multiple lines
    lines = ax.get_lines()
    assert len(lines) >= 3  # 2 data lines + chance line

    # Should have legend
    assert ax.get_legend() is not None

    plt.close(fig)


def test_save_temporal_plot():
    """Test saving plot to file."""
    from scripts.visualize_temporal import save_temporal_plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_plot.png"
        save_temporal_plot(fig, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    plt.close(fig)


def test_temporal_csv_validation():
    """Test that temporal CSV has required columns."""
    from scripts.visualize_temporal import validate_temporal_csv

    # Valid CSV
    valid_df = pd.DataFrame({
        "ClassA": [11],
        "ClassB": [22],
        "Seed": [42],
        "TimeWindow_Center": [25],
        "Accuracy": [55.0],
    })

    assert validate_temporal_csv(valid_df) is True

    # Missing required column
    invalid_df = pd.DataFrame({
        "ClassA": [11],
        "Seed": [42],
        "Accuracy": [55.0],
    })

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_temporal_csv(invalid_df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
