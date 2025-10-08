"""
Tests for ordinal adjacent-pair decision layer (post-hoc refinement).

These tests define the expected behavior before implementation (TDD).
They encode all scientific requirements and guardrails from the Consultant's feedback.

Constitutional compliance:
- Section III (Deterministic Training): Leak-free θ tuning on inner data only
- Section IV (Rigorous Validation): No leakage into outer test during tuning
- Section V (Audit-Ready Artifacts): All decisions traceable and reproducible
"""

from __future__ import annotations
import json
import numpy as np
import pytest
from pathlib import Path
from typing import List, Dict

# Module under test (will be created after tests pass)
# from code.posthoc.decision_layer import (
#     are_adjacent,
#     normalize_probs,
#     apply_ratio_rule,
#     tune_theta,
#     apply_decision_rule_to_rows,
# )


class TestAdjacencyDetection:
    """Test adjacency detection based on numeric ordinal values."""
    
    def test_adjacent_indices_differ_by_one(self):
        """Indices whose numeric values differ by 1 are adjacent."""
        from code.posthoc.decision_layer import are_adjacent, build_ordinal_mapping
        
        ordinal_map = build_ordinal_mapping(["1", "2", "3"])  # {0: 1, 1: 2, 2: 3}
        
        assert are_adjacent(0, 1, ordinal_map) is True  # 1 and 2 are adjacent
        assert are_adjacent(1, 0, ordinal_map) is True  # Order doesn't matter
        assert are_adjacent(1, 2, ordinal_map) is True  # 2 and 3 are adjacent
        assert are_adjacent(2, 1, ordinal_map) is True
    
    def test_non_adjacent_indices(self):
        """Indices whose numeric values differ by >1 are not adjacent."""
        from code.posthoc.decision_layer import are_adjacent, build_ordinal_mapping
        
        ordinal_map = build_ordinal_mapping(["1", "2", "3"])
        
        assert are_adjacent(0, 2, ordinal_map) is False  # 1 and 3 are not adjacent
        assert are_adjacent(2, 0, ordinal_map) is False
    
    def test_same_index_not_adjacent(self):
        """Same index is not adjacent to itself."""
        from code.posthoc.decision_layer import are_adjacent, build_ordinal_mapping
        
        ordinal_map = build_ordinal_mapping(["1", "2", "3"])
        
        assert are_adjacent(0, 0, ordinal_map) is False
        assert are_adjacent(1, 1, ordinal_map) is False


class TestProbabilityNormalization:
    """Test probability vector normalization."""
    
    def test_already_normalized(self):
        """Probabilities that already sum to 1.0 remain unchanged."""
        from code.posthoc.decision_layer import normalize_probs
        
        probs = np.array([0.33, 0.33, 0.34])
        normalized = normalize_probs(probs)
        
        assert np.isclose(normalized.sum(), 1.0)
        np.testing.assert_array_almost_equal(normalized, probs)
    
    def test_floating_point_error(self):
        """Handle floating-point errors from ensemble averaging."""
        from code.posthoc.decision_layer import normalize_probs
        
        # Simulate ensemble average with tiny error
        probs = np.array([0.401, 0.399, 0.199])  # sum = 0.999
        normalized = normalize_probs(probs)
        
        assert np.isclose(normalized.sum(), 1.0, atol=1e-9)
        # Relative proportions preserved
        assert np.argmax(normalized) == 0
    
    def test_unnormalized_input(self):
        """Properly normalize unnormalized probabilities."""
        from code.posthoc.decision_layer import normalize_probs
        
        probs = np.array([2.0, 3.0, 5.0])  # sum = 10.0
        normalized = normalize_probs(probs)
        
        expected = np.array([0.2, 0.3, 0.5])
        np.testing.assert_array_almost_equal(normalized, expected)


class TestRatioRule:
    """Test the adjacent-pair ratio rule logic."""
    
    def test_adjacent_pair_below_threshold_predicts_lower(self):
        """When r < θ, predict lower index."""
        from code.posthoc.decision_layer import apply_ratio_rule, build_ordinal_mapping
        
        # Top-2 are indices 0 and 1 (adjacent: 1 and 2)
        # p(0)=0.52, p(1)=0.38, p(2)=0.10
        # r = 0.38/(0.52+0.38) = 0.422
        probs = np.array([0.52, 0.38, 0.10])
        theta = 0.50  # r < theta
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        pred = apply_ratio_rule(probs, theta, ordinal_map)
        
        assert pred == 0  # Lower index
    
    def test_adjacent_pair_above_threshold_predicts_upper(self):
        """When r >= θ, predict upper index."""
        from code.posthoc.decision_layer import apply_ratio_rule, build_ordinal_mapping
        
        # Top-2 are indices 0 and 1 (adjacent: 1 and 2)
        # p(0)=0.42, p(1)=0.48, p(2)=0.10
        # r = 0.48/(0.42+0.48) = 0.533
        probs = np.array([0.42, 0.48, 0.10])
        theta = 0.50  # r >= theta
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        pred = apply_ratio_rule(probs, theta, ordinal_map)
        
        assert pred == 1  # Upper index
    
    def test_non_adjacent_returns_argmax(self):
        """When top-2 are not adjacent, return standard argmax."""
        from code.posthoc.decision_layer import apply_ratio_rule, build_ordinal_mapping
        
        # Top-2 are indices 0 and 2 (NOT adjacent: 1 and 3)
        # Argmax = 2
        probs = np.array([0.35, 0.10, 0.55])
        theta = 0.50
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        pred = apply_ratio_rule(probs, theta, ordinal_map)
        
        assert pred == 2  # Argmax unchanged
    
    def test_ratio_at_exact_threshold(self):
        """When r exactly equals θ, predict upper (>= condition)."""
        from code.posthoc.decision_layer import apply_ratio_rule, build_ordinal_mapping
        
        # Construct so r = exactly 0.50
        # p(0)=0.50, p(1)=0.50, r = 0.50/(0.50+0.50) = 0.50
        probs = np.array([0.50, 0.50, 0.0])
        theta = 0.50
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        pred = apply_ratio_rule(probs, theta, ordinal_map)
        
        assert pred == 1  # Upper index (>= condition)
    
    def test_handles_unnormalized_probs(self):
        """Rule normalizes before computing ratio."""
        from code.posthoc.decision_layer import apply_ratio_rule, build_ordinal_mapping
        
        # Unnormalized (sum=0.999 due to floating-point)
        probs = np.array([0.42, 0.38, 0.199])
        theta = 0.50
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        # Should normalize then compute ratio correctly
        pred = apply_ratio_rule(probs, theta, ordinal_map)
        
        # r ≈ 0.38/(0.42+0.38) = 0.475 < 0.50 → predict lower
        assert pred == 0


class TestThetaTuning:
    """Test θ tuning on inner validation data."""
    
    def test_sweeps_grid_and_picks_best(self):
        """Tune θ by sweeping grid and picking best objective score."""
        from code.posthoc.decision_layer import tune_theta, build_ordinal_mapping
        from code.training.metrics import ObjectiveComputer
        
        # Synthetic inner predictions with known optimal θ
        # Construct so θ=0.55 maximizes macro-F1
        inner_rows = self._create_synthetic_inner_rows()
        
        cfg = {
            "optuna_objective": "inner_mean_macro_f1",
        }
        objective_computer = ObjectiveComputer(cfg)
        
        theta_grid = np.arange(0.30, 0.71, 0.05)
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        best_theta, stats = tune_theta(
            inner_rows=inner_rows,
            objective_computer=objective_computer,
            theta_grid=theta_grid,
            class_names=class_names,
            ordinal_map=ordinal_map,
            min_activation=10,  # Low threshold for test
        )
        
        # Should find a reasonable theta in the grid
        assert 0.30 <= best_theta <= 0.70
        assert stats["fallback"] is False
        assert stats["n_adjacent"] >= 10
    
    def test_insufficient_activations_returns_fallback(self):
        """When too few adjacent pairs, return θ=0.50 fallback."""
        from code.posthoc.decision_layer import tune_theta, build_ordinal_mapping
        from code.training.metrics import ObjectiveComputer
        
        # Only 5 trials, all non-adjacent (distant confusions)
        inner_rows = [
            {
                "outer_fold": 1,
                "inner_fold": 1,
                "trial_index": i,
                "subject_id": 2,
                "true_label_idx": 0,
                "true_label_name": "1",
                "pred_label_idx": 2,  # 0→2 confusion (not adjacent)
                "pred_label_name": "3",
                "correct": 0,
                "probs": json.dumps([0.30, 0.10, 0.60]),  # Top-2: 2 and 0 (not adjacent)
            }
            for i in range(5)
        ]
        
        cfg = {"optuna_objective": "inner_mean_macro_f1"}
        objective_computer = ObjectiveComputer(cfg)
        theta_grid = np.arange(0.30, 0.71, 0.05)
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        best_theta, stats = tune_theta(
            inner_rows=inner_rows,
            objective_computer=objective_computer,
            theta_grid=theta_grid,
            class_names=class_names,
            ordinal_map=ordinal_map,
            min_activation=10,  # Require 10, have <10
        )
        
        assert best_theta == 0.50  # Fallback
        assert stats["fallback"] is True
        assert stats["reason"] == "insufficient_activations"
        assert stats["n_adjacent"] < 10
    
    def test_uses_configured_objective(self):
        """θ tuning uses the same objective as hyperparameter search."""
        from code.posthoc.decision_layer import tune_theta, build_ordinal_mapping
        from code.training.metrics import ObjectiveComputer
        
        inner_rows = self._create_synthetic_inner_rows()
        
        # Test with composite objective (matching user's config)
        cfg = {
            "optuna_objective": "composite_min_f1_plur_corr",
            "min_f1_threshold": 38.0,
        }
        objective_computer = ObjectiveComputer(cfg)
        
        theta_grid = np.arange(0.40, 0.61, 0.05)
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        best_theta, stats = tune_theta(
            inner_rows=inner_rows,
            objective_computer=objective_computer,
            theta_grid=theta_grid,
            class_names=class_names,
            ordinal_map=ordinal_map,
            min_activation=10,
        )
        
        # Should successfully tune using composite metric
        assert 0.40 <= best_theta <= 0.60
        assert stats["fallback"] is False
        assert "best_score" in stats
    
    @staticmethod
    def _create_synthetic_inner_rows() -> List[Dict]:
        """Create synthetic inner validation predictions for testing."""
        np.random.seed(42)
        rows = []
        
        for i in range(100):
            true_idx = i % 3
            # Create adjacent confusions (0↔1, 1↔2)
            if i % 2 == 0:
                pred_idx = true_idx
                probs = [0.0, 0.0, 0.0]
                probs[true_idx] = 0.80
                probs[(true_idx + 1) % 3] = 0.15
                probs[(true_idx + 2) % 3] = 0.05
            else:
                # Adjacent confusion
                pred_idx = (true_idx + 1) % 3
                probs = [0.0, 0.0, 0.0]
                probs[pred_idx] = 0.48
                probs[true_idx] = 0.42
                probs[(true_idx + 2) % 3] = 0.10
            
            rows.append({
                "outer_fold": 1,
                "inner_fold": 1,
                "trial_index": i,
                "subject_id": (i % 5) + 2,
                "true_label_idx": true_idx,
                "true_label_name": str(true_idx + 1),
                "pred_label_idx": pred_idx,
                "pred_label_name": str(pred_idx + 1),
                "correct": int(pred_idx == true_idx),
                "probs": json.dumps(probs),
            })
        
        return rows


class TestApplyDecisionRuleToRows:
    """Test applying the decision rule to a list of prediction rows."""
    
    def test_modifies_only_adjacent_pairs(self):
        """Rule fires only when top-2 are adjacent."""
        from code.posthoc.decision_layer import apply_decision_rule_to_rows, build_ordinal_mapping
        
        rows = [
            # Adjacent pair (0-1): will be adjusted
            {
                "outer_fold": 1,
                "trial_index": 0,
                "true_label_idx": 0,
                "true_label_name": "1",
                "pred_label_idx": 0,  # Argmax
                "pred_label_name": "1",
                "probs": json.dumps([0.42, 0.48, 0.10]),  # Top-2: 1,0 (adjacent)
            },
            # Non-adjacent (0-2): unchanged
            {
                "outer_fold": 1,
                "trial_index": 1,
                "true_label_idx": 0,
                "true_label_name": "1",
                "pred_label_idx": 2,  # Argmax
                "pred_label_name": "3",
                "probs": json.dumps([0.30, 0.10, 0.60]),  # Top-2: 2,0 (not adjacent)
            },
        ]
        
        theta = 0.50
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        modified = apply_decision_rule_to_rows(rows, theta, class_names, ordinal_map)
        
        # Row 0: r = 0.48/(0.42+0.48) = 0.533 >= 0.50 → predict 1 (changed)
        assert modified[0]["pred_label_idx"] == 1
        assert modified[0]["pred_label_name"] == "2"
        
        # Row 1: Non-adjacent, unchanged
        assert modified[1]["pred_label_idx"] == 2
        assert modified[1]["pred_label_name"] == "3"
    
    def test_updates_correct_flag(self):
        """Updates 'correct' flag when prediction changes."""
        from code.posthoc.decision_layer import apply_decision_rule_to_rows, build_ordinal_mapping
        
        rows = [
            {
                "outer_fold": 1,
                "trial_index": 0,
                "true_label_idx": 1,  # True: "2"
                "true_label_name": "2",
                "pred_label_idx": 0,  # Baseline pred: "1" (wrong)
                "pred_label_name": "1",
                "correct": 0,
                "probs": json.dumps([0.42, 0.48, 0.10]),  # Top-2: 1,0 adjacent
            },
        ]
        
        theta = 0.50
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        modified = apply_decision_rule_to_rows(rows, theta, class_names, ordinal_map)
        
        # r = 0.48/(0.42+0.48) = 0.533 >= 0.50 → predict 1 ("2")
        assert modified[0]["pred_label_idx"] == 1
        assert modified[0]["correct"] == 1  # Now correct!
    
    def test_preserves_other_fields(self):
        """All non-prediction fields remain unchanged."""
        from code.posthoc.decision_layer import apply_decision_rule_to_rows, build_ordinal_mapping
        
        rows = [
            {
                "outer_fold": 1,
                "trial_index": 42,
                "subject_id": 5,
                "true_label_idx": 0,
                "true_label_name": "1",
                "pred_label_idx": 1,
                "pred_label_name": "2",
                "correct": 0,
                "p_trueclass": 0.42,
                "logp_trueclass": -0.868,
                "probs": json.dumps([0.42, 0.48, 0.10]),
            },
        ]
        
        theta = 0.50
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        modified = apply_decision_rule_to_rows(rows, theta, class_names, ordinal_map)
        
        # Non-prediction fields preserved
        assert modified[0]["outer_fold"] == 1
        assert modified[0]["trial_index"] == 42
        assert modified[0]["subject_id"] == 5
        assert modified[0]["p_trueclass"] == 0.42
        assert modified[0]["logp_trueclass"] == -0.868
        # Probs unchanged (rule doesn't modify probabilities)
        assert modified[0]["probs"] == json.dumps([0.42, 0.48, 0.10])


class TestLabelOrderingEdgeCases:
    """Test label ordering and adjacency with numeric ordinal values."""
    
    def test_numeric_string_labels(self):
        """Standard case: class_names = ["1", "2", "3"]."""
        from code.posthoc.decision_layer import apply_ratio_rule, build_ordinal_mapping
        
        probs = np.array([0.42, 0.48, 0.10])
        theta = 0.50
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)  # {0: 1, 1: 2, 2: 3}
        
        pred = apply_ratio_rule(probs, theta, ordinal_map)
        
        # Indices 0-1 have numeric values 1-2 (adjacent)
        assert pred == 1
    
    def test_larger_range(self):
        """Works with 6-class task (4-9)."""
        from code.posthoc.decision_layer import apply_ratio_rule, build_ordinal_mapping
        
        # Top-2: indices 1 and 2 (adjacent by numeric values: 5 and 6)
        probs = np.array([0.05, 0.42, 0.48, 0.03, 0.01, 0.01])
        theta = 0.50
        class_names = ["4", "5", "6", "7", "8", "9"]
        ordinal_map = build_ordinal_mapping(class_names)  # {0: 4, 1: 5, 2: 6, ...}
        
        pred = apply_ratio_rule(probs, theta, ordinal_map)
        
        # r = 0.48/(0.42+0.48) = 0.533 >= 0.50 → predict upper (index 2)
        assert pred == 2


class TestProvenanceAndDeterminism:
    """Test that tuning and application are deterministic and auditable."""
    
    def test_same_inputs_produce_same_theta(self):
        """Determinism: same inputs → same θ."""
        from code.posthoc.decision_layer import tune_theta, build_ordinal_mapping
        from code.training.metrics import ObjectiveComputer
        
        inner_rows = TestThetaTuning._create_synthetic_inner_rows()
        cfg = {"optuna_objective": "inner_mean_macro_f1"}
        objective_computer = ObjectiveComputer(cfg)
        theta_grid = np.arange(0.30, 0.71, 0.05)
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        # Run twice
        theta1, stats1 = tune_theta(
            inner_rows, objective_computer, theta_grid, class_names, ordinal_map, min_activation=10
        )
        theta2, stats2 = tune_theta(
            inner_rows, objective_computer, theta_grid, class_names, ordinal_map, min_activation=10
        )
        
        assert theta1 == theta2
        assert stats1["n_adjacent"] == stats2["n_adjacent"]
    
    def test_stats_include_provenance(self):
        """Stats dict includes all necessary provenance fields."""
        from code.posthoc.decision_layer import tune_theta, build_ordinal_mapping
        from code.training.metrics import ObjectiveComputer
        
        inner_rows = TestThetaTuning._create_synthetic_inner_rows()
        cfg = {"optuna_objective": "inner_mean_macro_f1"}
        objective_computer = ObjectiveComputer(cfg)
        theta_grid = np.arange(0.30, 0.71, 0.05)
        class_names = ["1", "2", "3"]
        ordinal_map = build_ordinal_mapping(class_names)
        
        best_theta, stats = tune_theta(
            inner_rows, objective_computer, theta_grid, class_names, ordinal_map, min_activation=10
        )
        
        # Required fields
        assert "fallback" in stats
        assert "n_adjacent" in stats
        assert "n_inner_trials" in stats
        
        if not stats["fallback"]:
            assert "best_score" in stats


# Fixtures for integration tests (when artifact writer is implemented)

@pytest.fixture
def mock_inner_predictions():
    """Mock inner prediction rows for integration tests."""
    return TestThetaTuning._create_synthetic_inner_rows()


@pytest.fixture
def mock_outer_predictions():
    """Mock outer prediction rows for integration tests."""
    np.random.seed(42)
    rows = []
    
    for i in range(50):
        true_idx = i % 3
        pred_idx = (true_idx + (i % 2)) % 3
        probs = [0.33, 0.33, 0.34]
        probs[pred_idx] += 0.15
        
        rows.append({
            "outer_fold": 1,
            "trial_index": i,
            "subject_id": (i % 5) + 2,
            "true_label_idx": true_idx,
            "true_label_name": str(true_idx + 1),
            "pred_label_idx": pred_idx,
            "pred_label_name": str(pred_idx + 1),
            "correct": int(pred_idx == true_idx),
            "p_trueclass": probs[true_idx],
            "logp_trueclass": np.log(max(probs[true_idx], 1e-12)),
            "probs": json.dumps(probs),
        })
    
    return rows

