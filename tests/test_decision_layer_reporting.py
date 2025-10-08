"""
TDD tests for decision layer reporting enhancements.

Tests for:
- Baseline vs thresholded metrics computation
- Statistical tests (McNemar, paired t-test)
- TXT report formatting
- JSON summary fields
- Comparison plots generation

Constitutional compliance: Section III (Deterministic), Section V (Audit-Ready)
"""

import pytest
import numpy as np
import json
from pathlib import Path
from typing import List, Dict

# Module under test (will be extended)
from code.posthoc import decision_layer


class TestMetricsComputation:
    """Test baseline vs thresholded metrics computation."""
    
    def test_compute_metrics_from_rows(self):
        """Test that metrics are correctly computed from prediction rows."""
        # 3-class problem, 9 trials
        rows = [
            {"true_label_idx": 0, "pred_label_idx": 0, "probs": "[0.7, 0.2, 0.1]"},  # correct
            {"true_label_idx": 0, "pred_label_idx": 1, "probs": "[0.3, 0.5, 0.2]"},  # wrong
            {"true_label_idx": 0, "pred_label_idx": 0, "probs": "[0.6, 0.3, 0.1]"},  # correct
            {"true_label_idx": 1, "pred_label_idx": 1, "probs": "[0.2, 0.6, 0.2]"},  # correct
            {"true_label_idx": 1, "pred_label_idx": 1, "probs": "[0.1, 0.7, 0.2]"},  # correct
            {"true_label_idx": 1, "pred_label_idx": 2, "probs": "[0.1, 0.4, 0.5]"},  # wrong
            {"true_label_idx": 2, "pred_label_idx": 2, "probs": "[0.1, 0.2, 0.7]"},  # correct
            {"true_label_idx": 2, "pred_label_idx": 1, "probs": "[0.2, 0.5, 0.3]"},  # wrong
            {"true_label_idx": 2, "pred_label_idx": 2, "probs": "[0.1, 0.3, 0.6]"},  # correct
        ]
        
        metrics = decision_layer.compute_metrics_from_rows(rows, num_classes=3)
        
        # 6 correct out of 9
        assert abs(metrics["acc"] - 66.67) < 0.1
        assert "macro_f1" in metrics
        assert "min_per_class_f1" in metrics
        assert "plur_corr" in metrics
        assert "cohen_kappa" in metrics
        assert "per_class_f1" in metrics
        assert len(metrics["per_class_f1"]) == 3
    
    def test_compute_fold_comparison(self):
        """Test computing baseline vs thresholded metrics for one fold."""
        baseline_rows = [
            {"outer_fold": 1, "true_label_idx": 0, "pred_label_idx": 1, "probs": "[0.4, 0.5, 0.1]"},
            {"outer_fold": 1, "true_label_idx": 1, "pred_label_idx": 1, "probs": "[0.2, 0.6, 0.2]"},
            {"outer_fold": 1, "true_label_idx": 2, "pred_label_idx": 2, "probs": "[0.1, 0.3, 0.6]"},
        ]
        
        thresholded_rows = [
            {"outer_fold": 1, "true_label_idx": 0, "pred_label_idx": 0, "probs": "[0.4, 0.5, 0.1]"},  # fixed!
            {"outer_fold": 1, "true_label_idx": 1, "pred_label_idx": 1, "probs": "[0.2, 0.6, 0.2]"},
            {"outer_fold": 1, "true_label_idx": 2, "pred_label_idx": 2, "probs": "[0.1, 0.3, 0.6]"},
        ]
        
        comparison = decision_layer.compute_fold_comparison(
            baseline_rows=baseline_rows,
            thresholded_rows=thresholded_rows,
            fold=1,
            num_classes=3
        )
        
        assert comparison["fold"] == 1
        assert "baseline" in comparison
        assert "thresholded" in comparison
        assert "deltas" in comparison
        
        # Thresholded should be better (fixed one error)
        assert comparison["thresholded"]["acc"] > comparison["baseline"]["acc"]
        assert comparison["deltas"]["acc"] > 0


class TestStatisticalTests:
    """Test statistical significance tests."""
    
    def test_mcnemar_test(self):
        """Test McNemar test for paired predictions."""
        # Create baseline and thresholded predictions
        baseline_correct = [1, 0, 1, 0, 1, 1, 0, 0]
        thresholded_correct = [1, 1, 1, 0, 1, 1, 1, 0]  # Fixed 2 errors
        
        result = decision_layer.mcnemar_test(baseline_correct, thresholded_correct)
        
        assert "chi2" in result
        assert "p_value" in result
        assert result["chi2"] >= 0
        assert 0 <= result["p_value"] <= 1
    
    def test_paired_ttest(self):
        """Test paired t-test across folds."""
        # Per-fold accuracies
        baseline_accs = [45.2, 48.1, 42.3, 50.0, 46.5, 44.8]
        thresholded_accs = [48.5, 51.2, 45.1, 52.3, 49.0, 47.2]  # All improved
        
        result = decision_layer.paired_ttest(baseline_accs, thresholded_accs)
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert result["df"] == 5  # n - 1
        assert result["p_value"] < 0.05  # Should be significant


class TestTXTReportFormatting:
    """Test TXT report decision layer section formatting."""
    
    def test_format_decision_layer_section(self):
        """Test formatting of decision layer TXT section."""
        stats = {
            "baseline": {
                "overall": {"acc": 48.2, "std_acc": 3.1, "macro_f1": 47.8, 
                           "min_per_class_f1": 38.5, "plur_corr": 66.7, "cohen_kappa": 0.223},
            },
            "thresholded": {
                "overall": {"acc": 51.7, "std_acc": 2.9, "macro_f1": 51.2,
                           "min_per_class_f1": 42.1, "plur_corr": 100.0, "cohen_kappa": 0.276},
            },
            "deltas": {
                "acc": 3.5, "macro_f1": 3.4, "min_per_class_f1": 3.6,
                "plur_corr": 33.3, "cohen_kappa": 0.053,
            },
            "config": {
                "metric": "composite_min_f1_plur_corr",
                "theta_grid": [0.30, 0.70, 0.01],
                "min_activation_trials": 50,
            },
            "per_fold": [
                {"fold": 1, "theta": 0.48, "n_activated": 127, "n_outer_trials": 412, "activation_rate": 0.308},
                {"fold": 2, "theta": 0.52, "n_activated": 143, "n_outer_trials": 398, "activation_rate": 0.359},
            ],
            "statistical_tests": {
                "mcnemar_chi2": 14.7,
                "mcnemar_p": 0.0001,
                "paired_t_statistic": 4.2,
                "paired_t_p": 0.008,
            }
        }
        
        lines = decision_layer.format_decision_layer_txt_section(stats)
        
        # Check structure
        assert any("DECISION LAYER ANALYSIS" in line for line in lines)
        assert any("Baseline (Argmax)" in line for line in lines)
        assert any("Decision Layer (Ratio Rule)" in line for line in lines)
        assert any("Per-Fold Thresholds" in line for line in lines)
        assert any("Statistical Comparison" in line for line in lines)
        
        # Check values appear
        txt = "\n".join(lines)
        assert "48.2%" in txt  # baseline acc
        assert "51.7%" in txt  # thresholded acc
        assert "+3.5 pp" in txt  # delta
        assert "θ=0.48" in txt  # fold 1 theta


class TestJSONSummaryFields:
    """Test JSON summary decision layer fields."""
    
    def test_build_decision_layer_json_fields(self):
        """Test building decision layer JSON fields."""
        stats = {
            "baseline": {"overall": {"acc": 48.2, "macro_f1": 47.8}},
            "thresholded": {"overall": {"acc": 51.7, "macro_f1": 51.2}},
            "deltas": {"acc": 3.5, "macro_f1": 3.4},
            "config": {"metric": "composite_min_f1_plur_corr"},
            "per_fold": [
                {"fold": 1, "theta": 0.48, "activation_rate": 0.308},
                {"fold": 2, "theta": 0.52, "activation_rate": 0.359},
            ],
            "overall_activation_rate": 0.345,
            "total_activated_trials": 847,
            "total_test_trials": 2464,
            "statistical_tests": {"mcnemar_chi2": 14.7, "mcnemar_p": 0.0001},
        }
        
        json_fields = decision_layer.build_decision_layer_json_fields(stats)
        
        # Check consultant schema
        assert json_fields["mean_acc"] == 51.7
        assert json_fields["mean_acc_argmax_baseline"] == 48.2
        assert json_fields["mean_acc_delta"] == 3.5
        
        assert "decision_layer" in json_fields
        dl = json_fields["decision_layer"]
        assert dl["enabled"] == True
        assert dl["metric_optimized"] == "composite_min_f1_plur_corr"
        assert dl["overall_activation_rate"] == 0.345
        assert "per_fold_thetas" in dl
        assert "statistical_tests" in dl


class TestComparisonPlots:
    """Test comparison plot generation."""
    
    def test_create_confusion_comparison_plot(self, tmp_path):
        """Test side-by-side confusion matrix comparison."""
        # Create mock predictions
        baseline_rows = [
            {"true_label_idx": i % 3, "pred_label_idx": (i + 1) % 3, "probs": "[0.3, 0.4, 0.3]"}
            for i in range(30)
        ]
        
        thresholded_rows = [
            {"true_label_idx": i % 3, "pred_label_idx": i % 3, "probs": "[0.3, 0.4, 0.3]"}  # All correct
            for i in range(30)
        ]
        
        out_dir = tmp_path / "plots_outer_threshold_compare"
        out_dir.mkdir()
        
        decision_layer.create_confusion_comparison_plot(
            baseline_rows=baseline_rows,
            thresholded_rows=thresholded_rows,
            class_names=["1", "2", "3"],
            out_path=out_dir / "confusion_comparison.png",
            title="Overall Confusion: Baseline vs Thresholded"
        )
        
        # Check file was created
        assert (out_dir / "confusion_comparison.png").exists()
    
    def test_create_per_fold_comparison_bars(self, tmp_path):
        """Test per-fold accuracy comparison bar chart."""
        fold_comparisons = [
            {"fold": 1, "baseline": {"acc": 45.2}, "thresholded": {"acc": 48.5}},
            {"fold": 2, "baseline": {"acc": 48.1}, "thresholded": {"acc": 51.2}},
            {"fold": 3, "baseline": {"acc": 42.3}, "thresholded": {"acc": 45.1}},
        ]
        
        out_dir = tmp_path / "plots_outer_threshold_compare"
        out_dir.mkdir()
        
        decision_layer.create_per_fold_comparison_bars(
            fold_comparisons=fold_comparisons,
            out_path=out_dir / "per_fold_accuracy_comparison.png"
        )
        
        assert (out_dir / "per_fold_accuracy_comparison.png").exists()

