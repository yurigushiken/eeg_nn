"""
Integration test for the complete XAI pipeline.

Tests the end-to-end flow from loading a completed run to generating
all XAI artifacts including:
- Per-fold IG attributions and heatmaps
- Per-fold Grad-CAM heatmaps
- Grand-average IG attributions and visualizations
- Per-class grand-average IG visualizations
- Time-frequency analysis
- Top-2 spatio-temporal event topomaps
- Consolidated HTML report

This test MUST FAIL initially (TDD) and only pass once implementation is complete.
"""

import pytest
import subprocess
import sys
from pathlib import Path
from glob import glob


@pytest.fixture
def fixture_dir():
    """Return path to the test fixture directory."""
    return Path(__file__).parent.parent / "fixtures" / "xai_test_run"


@pytest.fixture
def run_xai_script():
    """Return path to run_xai_analysis.py script."""
    return Path(__file__).parent.parent.parent / "scripts" / "run_xai_analysis.py"


def test_xai_fixture_exists(fixture_dir):
    """Verify that the test fixture directory exists."""
    assert fixture_dir.exists(), f"Test fixture not found at {fixture_dir}"
    assert (fixture_dir / "summary_test_xai_fixture_001.json").exists(), \
        "summary.json not found in fixture"


def test_xai_pipeline_generates_per_fold_ig_artifacts(fixture_dir, run_xai_script):
    """
    Test that running XAI analysis generates per-fold IG artifacts.
    
    Expected outputs for fold 01:
    - xai_analysis/integrated_gradients/fold_01_xai_attributions.npy
    - xai_analysis/integrated_gradients/fold_01_xai_heatmap.png
    """
    # Run the XAI analysis script
    result = subprocess.run(
        [sys.executable, str(run_xai_script), "--run-dir", str(fixture_dir)],
        capture_output=True,
        text=True
    )
    
    # Check that script executed successfully
    assert result.returncode == 0, \
        f"XAI script failed with error:\n{result.stderr}"
    
    # Check per-fold IG artifacts exist
    xai_dir = fixture_dir / "xai_analysis"
    ig_dir = xai_dir / "integrated_gradients"
    
    assert ig_dir.exists(), "integrated_gradients directory must exist"
    
    # Check fold 01
    assert (ig_dir / "fold_01_xai_attributions.npy").exists(), \
        "fold_01_xai_attributions.npy must exist"
    assert (ig_dir / "fold_01_xai_heatmap.png").exists(), \
        "fold_01_xai_heatmap.png must exist"


def test_xai_pipeline_generates_per_fold_class_labels(fixture_dir):
    """
    Test that per-fold class labels are saved for per-class filtering.
    
    Expected output:
    - xai_analysis/integrated_gradients_per_class/fold_01_class_labels.npy
    """
    xai_dir = fixture_dir / "xai_analysis"
    ig_class_dir = xai_dir / "integrated_gradients_per_class"
    
    assert ig_class_dir.exists(), \
        "integrated_gradients_per_class directory must exist"
    
    # Note: The exact filename may vary; checking for pattern
    class_label_files = list(ig_class_dir.glob("fold_01_class_*.npy"))
    assert len(class_label_files) > 0, \
        "At least one per-class file must exist for fold 01"


def test_xai_pipeline_generates_per_fold_gradcam(fixture_dir):
    """
    Test that per-fold Grad-CAM heatmaps are generated.
    
    Expected output:
    - xai_analysis/gradcam_heatmaps/fold_01_gradcam_heatmap.png
    """
    xai_dir = fixture_dir / "xai_analysis"
    gradcam_dir = xai_dir / "gradcam_heatmaps"
    
    assert gradcam_dir.exists(), "gradcam_heatmaps directory must exist"
    assert (gradcam_dir / "fold_01_gradcam_heatmap.png").exists(), \
        "fold_01_gradcam_heatmap.png must exist"


def test_xai_pipeline_generates_grand_average_ig(fixture_dir):
    """
    Test that grand-average IG artifacts are generated.
    
    Expected outputs:
    - xai_analysis/grand_average_xai_attributions.npy
    - xai_analysis/grand_average_xai_heatmap.png
    - xai_analysis/grand_average_xai_topoplot.png
    """
    xai_dir = fixture_dir / "xai_analysis"
    
    assert (xai_dir / "grand_average_xai_attributions.npy").exists(), \
        "grand_average_xai_attributions.npy must exist"
    assert (xai_dir / "grand_average_xai_heatmap.png").exists(), \
        "grand_average_xai_heatmap.png must exist"
    assert (xai_dir / "grand_average_xai_topoplot.png").exists(), \
        "grand_average_xai_topoplot.png must exist"


def test_xai_pipeline_generates_per_class_grand_average(fixture_dir):
    """
    Test that per-class grand-average IG visualizations are generated.
    
    Expected outputs (pattern):
    - xai_analysis/grand_average_per_class/class_00_*_xai_heatmap.png
    """
    xai_dir = fixture_dir / "xai_analysis"
    per_class_dir = xai_dir / "grand_average_per_class"
    
    assert per_class_dir.exists(), \
        "grand_average_per_class directory must exist"
    
    # Check for at least one per-class heatmap
    class_heatmaps = list(per_class_dir.glob("class_00_*_xai_heatmap.png"))
    assert len(class_heatmaps) > 0, \
        "At least one per-class heatmap must exist"


def test_xai_pipeline_generates_time_frequency_analysis(fixture_dir):
    """
    Test that time-frequency analysis is generated.
    
    Expected output:
    - xai_analysis/grand_average_time_frequency.png
    """
    xai_dir = fixture_dir / "xai_analysis"
    
    assert (xai_dir / "grand_average_time_frequency.png").exists(), \
        "grand_average_time_frequency.png must exist"


def test_xai_pipeline_generates_peak_topomaps(fixture_dir):
    """
    Test that Top-2 spatio-temporal event topomaps are generated.
    
    Expected outputs (pattern):
    - xai_analysis/grand_average_ig_peak1_topoplot_*.png
    - xai_analysis/grand_average_ig_peak2_topoplot_*.png
    """
    xai_dir = fixture_dir / "xai_analysis"
    
    # Check for peak 1 topoplot
    peak1_files = list(xai_dir.glob("grand_average_ig_peak1_topoplot_*.png"))
    assert len(peak1_files) > 0, \
        "At least one peak1 topoplot must exist"
    
    # Check for peak 2 topoplot
    peak2_files = list(xai_dir.glob("grand_average_ig_peak2_topoplot_*.png"))
    assert len(peak2_files) > 0, \
        "At least one peak2 topoplot must exist"


def test_xai_pipeline_generates_consolidated_report(fixture_dir):
    """
    Test that consolidated HTML report is generated.
    
    Expected output:
    - consolidated_xai_report.html
    """
    report_path = fixture_dir / "consolidated_xai_report.html"
    
    assert report_path.exists(), \
        "consolidated_xai_report.html must exist"
    
    # Basic content validation
    content = report_path.read_text(encoding="utf-8")
    assert "XAI Report" in content, \
        "Report must contain 'XAI Report' title"
    assert "Grand Average" in content or "grand average" in content, \
        "Report must mention grand average"


def test_xai_pipeline_all_artifacts_present(fixture_dir):
    """
    Comprehensive test that all expected XAI artifacts are present.
    
    This is the master integration test that verifies the complete pipeline.
    """
    xai_dir = fixture_dir / "xai_analysis"
    
    # All required artifacts
    required_artifacts = [
        # Per-fold IG
        "integrated_gradients/fold_01_xai_attributions.npy",
        "integrated_gradients/fold_01_xai_heatmap.png",
        # Per-fold Grad-CAM
        "gradcam_heatmaps/fold_01_gradcam_heatmap.png",
        # Grand-average IG
        "grand_average_xai_attributions.npy",
        "grand_average_xai_heatmap.png",
        "grand_average_xai_topoplot.png",
        # Time-frequency
        "grand_average_time_frequency.png",
    ]
    
    missing_artifacts = []
    for artifact in required_artifacts:
        full_path = xai_dir / artifact
        if not full_path.exists():
            missing_artifacts.append(artifact)
    
    # Pattern-based checks (wildcards)
    pattern_checks = [
        ("integrated_gradients_per_class/fold_01_class_*.npy", 
         "Per-fold per-class files"),
        ("grand_average_per_class/class_00_*_xai_heatmap.png",
         "Per-class grand-average heatmaps"),
        ("grand_average_ig_peak1_topoplot_*.png",
         "Peak 1 topomaps"),
        ("grand_average_ig_peak2_topoplot_*.png",
         "Peak 2 topomaps"),
    ]
    
    pattern_failures = []
    for pattern, description in pattern_checks:
        matches = list(xai_dir.glob(pattern))
        if not matches:
            pattern_failures.append(f"{description} (pattern: {pattern})")
    
    # Consolidated report
    if not (fixture_dir / "consolidated_xai_report.html").exists():
        missing_artifacts.append("../consolidated_xai_report.html")
    
    # Compile error message
    error_parts = []
    if missing_artifacts:
        error_parts.append(f"Missing artifacts:\n  - " + "\n  - ".join(missing_artifacts))
    if pattern_failures:
        error_parts.append(f"Missing pattern matches:\n  - " + "\n  - ".join(pattern_failures))
    
    assert not error_parts, "\n\n".join(error_parts)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

