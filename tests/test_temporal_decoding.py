"""
Test suite for time-resolved RSA temporal decoding.

This module validates the temporal windowing logic, configuration handling,
and result aggregation for the sliding-window RSA approach.
"""
import math
from pathlib import Path

import pytest
import yaml


def test_temporal_window_generation():
    """Test that temporal windows are generated correctly."""
    from scripts.run_temporal_decoding import generate_temporal_windows

    # Standard parameters: 500ms epoch, 50ms window, 20ms stride
    windows = generate_temporal_windows(epoch_ms=500, window_ms=50, stride_ms=20)

    # Should generate 23 windows
    assert len(windows) == 23, f"Expected 23 windows, got {len(windows)}"

    # First window should start at 0
    assert windows[0] == (0, 50), f"First window should be (0, 50), got {windows[0]}"

    # Last window (window 23: starts at 440, ends at 490)
    # 440 + 50 = 490, which is < 500, so this is the last valid window
    assert windows[-1] == (440, 490), f"Last window should be (440, 490), got {windows[-1]}"

    # All windows should be exactly 50ms
    for start, end in windows:
        assert end - start == 50, f"Window ({start}, {end}) is not 50ms wide"

    # Windows should have 20ms stride
    for i in range(len(windows) - 1):
        stride = windows[i + 1][0] - windows[i][0]
        assert stride == 20, f"Stride between window {i} and {i+1} is {stride}, expected 20"

    # No gaps or overlaps beyond expected stride
    for i in range(len(windows) - 1):
        gap = windows[i + 1][0] - windows[i][1]
        assert gap == -30, f"Gap between windows should be -30ms (75% overlap), got {gap}"


def test_temporal_window_generation_edge_cases():
    """Test edge cases for window generation."""
    from scripts.run_temporal_decoding import generate_temporal_windows

    # Exact fit (no partial window at end)
    windows = generate_temporal_windows(epoch_ms=100, window_ms=50, stride_ms=50)
    assert len(windows) == 2
    assert windows == [(0, 50), (50, 100)]

    # Single window case
    windows = generate_temporal_windows(epoch_ms=50, window_ms=50, stride_ms=20)
    assert len(windows) == 1
    assert windows == [(0, 50)]

    # Very small stride (high overlap)
    windows = generate_temporal_windows(epoch_ms=100, window_ms=50, stride_ms=10)
    assert len(windows) == 6
    assert windows[0] == (0, 50)
    assert windows[-1] == (50, 100)


def test_temporal_window_naming():
    """Test that temporal window names are formatted correctly."""
    from scripts.run_temporal_decoding import format_window_name

    # Standard naming
    assert format_window_name(0, 50) == "t0-50ms"
    assert format_window_name(100, 150) == "t100-150ms"
    assert format_window_name(450, 500) == "t450-500ms"

    # Edge cases
    assert format_window_name(0, 100) == "t0-100ms"
    assert format_window_name(200, 250) == "t200-250ms"


def test_temporal_config_modification():
    """Test that config is properly modified with crop_ms for each window."""
    from scripts.run_temporal_decoding import apply_temporal_window_to_config

    base_cfg = {
        'seed': 42,
        'epochs': 100,
        'crop_ms': None,  # Will be overwritten
        'model_name': 'eegnex',
    }

    # Apply window (100, 150)
    windowed_cfg = apply_temporal_window_to_config(base_cfg, 100, 150)

    # Original config should be unchanged
    assert base_cfg['crop_ms'] is None

    # Windowed config should have crop_ms set
    assert windowed_cfg['crop_ms'] == [100, 150]

    # Other parameters should be preserved
    assert windowed_cfg['seed'] == 42
    assert windowed_cfg['epochs'] == 100
    assert windowed_cfg['model_name'] == 'eegnex'


def test_temporal_results_aggregation(tmp_path: Path):
    """Test that temporal results are aggregated correctly with time metadata."""
    from scripts.run_temporal_decoding import aggregate_temporal_metrics

    run_dir = Path(tmp_path)
    csv_path = run_dir / "outer_eval_metrics.csv"
    csv_path.write_text(
        "outer_fold,acc,macro_f1,min_per_class_f1\n"
        "1,60.0,58.0,55.0\n"
        "OVERALL,62.5,60.5,57.0\n"
    )

    # Aggregate with temporal metadata
    rows = list(aggregate_temporal_metrics(
        run_dir,
        class_a=11,
        class_b=33,
        seed=42,
        window_start=100,
        window_end=150
    ))

    assert len(rows) == 1
    assert rows[0] == {
        "ClassA": 11,
        "ClassB": 33,
        "Seed": 42,
        "TimeWindow_Start": 100,
        "TimeWindow_End": 150,
        "TimeWindow_Center": 125,
        "Accuracy": 62.5,
        "MacroF1": 60.5,
        "MinClassF1": 57.0,
    }


def test_temporal_csv_header():
    """Test that temporal CSV has correct header with time columns."""
    from scripts.run_temporal_decoding import get_temporal_csv_header

    header = get_temporal_csv_header()

    # Should include all standard RSA columns plus temporal metadata
    assert "ClassA" in header
    assert "ClassB" in header
    assert "Seed" in header
    assert "Accuracy" in header
    assert "MacroF1" in header
    assert "MinClassF1" in header

    # Temporal-specific columns
    assert "TimeWindow_Start" in header
    assert "TimeWindow_End" in header
    assert "TimeWindow_Center" in header


def test_temporal_run_directory_naming():
    """Test that temporal run directories include window information."""
    from scripts.run_temporal_decoding import format_temporal_run_dir

    # Standard naming pattern
    run_dir = format_temporal_run_dir(
        launch_id="20250101_120000",
        class_a=11,
        class_b=22,
        seed=42,
        window_start=100,
        window_end=150
    )

    assert "20250101_120000" in run_dir
    assert "rsa_11v22" in run_dir
    assert "seed_42" in run_dir
    assert "t100-150ms" in run_dir

    # Should be parseable back
    assert run_dir == "20250101_120000_rsa_11v22_seed_42_t100-150ms"


def test_temporal_state_tracking():
    """Test that temporal state includes window tracking."""
    from scripts.run_temporal_decoding import build_temporal_state

    pairs = [(11, 22), (33, 44)]
    seeds = [42, 123]
    windows = [(0, 50), (20, 70), (40, 90)]

    state = build_temporal_state(
        launch_id="test_launch",
        pairs=pairs,
        seeds=seeds,
        windows=windows,
        completed=[]
    )

    assert state["launch_id"] == "test_launch"
    assert state["pairs"] == pairs
    assert state["seeds"] == seeds
    assert state["windows"] == windows
    assert state["total_runs"] == 2 * 2 * 3  # pairs × seeds × windows = 12
    assert state["completed"] == []
    assert state["finished"] is False


def test_temporal_completion_tracking():
    """Test that completion tracking handles 3D space (pair, seed, window)."""
    from scripts.run_temporal_decoding import is_run_completed, mark_run_completed

    state = {
        "completed": [
            {"ClassA": 11, "ClassB": 22, "Seed": 42, "TimeWindow_Start": 0, "TimeWindow_End": 50},
            {"ClassA": 11, "ClassB": 22, "Seed": 42, "TimeWindow_Start": 20, "TimeWindow_End": 70},
        ]
    }

    # Completed run should return True
    assert is_run_completed(state, 11, 22, 42, 0, 50) is True
    assert is_run_completed(state, 11, 22, 42, 20, 70) is True

    # Non-completed run should return False
    assert is_run_completed(state, 11, 22, 42, 40, 90) is False
    assert is_run_completed(state, 33, 44, 42, 0, 50) is False

    # Mark new run as completed
    new_state = mark_run_completed(state, 11, 22, 42, 40, 90)
    assert is_run_completed(new_state, 11, 22, 42, 40, 90) is True
    assert len(new_state["completed"]) == 3


def test_temporal_window_sample_calculation():
    """Test conversion from ms to samples at 250Hz."""
    from scripts.run_temporal_decoding import ms_to_samples, samples_to_ms

    # 50ms window at 250Hz should be ~12-13 samples
    samples = ms_to_samples(50, sfreq=250)
    assert samples == 12 or samples == 13, f"50ms should be 12-13 samples at 250Hz, got {samples}"

    # 20ms stride at 250Hz should be 5 samples
    samples = ms_to_samples(20, sfreq=250)
    assert samples == 5, f"20ms should be 5 samples at 250Hz, got {samples}"

    # 500ms epoch at 250Hz should be 125 samples
    samples = ms_to_samples(500, sfreq=250)
    assert samples == 125, f"500ms should be 125 samples at 250Hz, got {samples}"

    # Round-trip conversion
    assert samples_to_ms(125, sfreq=250) == 500.0
    assert samples_to_ms(12, sfreq=250) == 48.0


def _eegnex_time_dims(n_times: int, pool4: int, pool5: int) -> tuple[int, int]:
    """Replicate EEGNeX temporal downsampling to ensure pooling is valid."""
    t3 = math.floor((n_times + 2 - pool4) / pool4) + 1
    t5 = math.floor((t3 + 2 - pool5) / pool5) + 1
    return t3, t5


def test_temporal_pooling_supports_50ms_window():
    """Ensure pooling factors keep time dimension > 0 for 50ms windows."""
    from scripts.run_temporal_decoding import ms_to_samples

    cfg = yaml.safe_load(Path("configs/tasks/rsa_temporal.yaml").read_text())
    pool4 = int(cfg.get("avg_pool_block4", 4))
    pool5 = int(cfg.get("avg_pool_block5", 8))
    window_ms = cfg["temporal"]["window_ms"]
    n_times = ms_to_samples(window_ms, sfreq=250)
    t3, t5 = _eegnex_time_dims(n_times, pool4, pool5)

    assert t3 >= 1, f"Block4 pooling collapsed time dim: t3={t3}"
    assert t5 >= 1, f"Block5 pooling collapsed time dim: t5={t5}"


def test_temporal_results_sorting():
    """Test that temporal results are sorted by time, then pair, then seed."""
    from scripts.run_temporal_decoding import sort_temporal_results

    results = [
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "TimeWindow_Start": 100, "Accuracy": 60.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 42, "TimeWindow_Start": 0, "Accuracy": 55.0},
        {"ClassA": 11, "ClassB": 22, "Seed": 123, "TimeWindow_Start": 0, "Accuracy": 58.0},
        {"ClassA": 11, "ClassB": 33, "Seed": 42, "TimeWindow_Start": 0, "Accuracy": 62.0},
    ]

    sorted_results = sort_temporal_results(results)

    # Should be sorted by: TimeWindow_Start, ClassA, ClassB, Seed
    assert sorted_results[0]["TimeWindow_Start"] == 0
    assert sorted_results[0]["ClassA"] == 11
    assert sorted_results[0]["ClassB"] == 22
    assert sorted_results[0]["Seed"] == 42

    assert sorted_results[-1]["TimeWindow_Start"] == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
