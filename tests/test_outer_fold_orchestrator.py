"""
Tests for OuterFoldOrchestrator class (training/outer_loop.py).

This class orchestrates one complete outer fold:
- Inner fold iteration
- Model/optimizer/loss setup per inner fold
- Inner training via InnerTrainer
- Inner result aggregation
- Best inner model selection
- Outer evaluation via OuterEvaluator
- Per-fold plotting
- Comprehensive result collection

Constitutional compliance: Section III (Deterministic Training), Section IV (Subject-Aware CV)
"""
import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path


def test_outer_fold_orchestrator_imports():
    """Test that OuterFoldOrchestrator can be imported (will fail pre-extraction)."""
    try:
        from code.training.outer_loop import OuterFoldOrchestrator
        assert OuterFoldOrchestrator is not None
    except ImportError as e:
        pytest.fail(f"Cannot import OuterFoldOrchestrator: {e}")


def test_outer_fold_orchestrator_init():
    """Test OuterFoldOrchestrator initialization."""
    from code.training.outer_loop import OuterFoldOrchestrator
    from code.training.metrics import ObjectiveComputer
    from code.training.inner_loop import InnerTrainer
    from code.training.evaluation import OuterEvaluator
    from code.artifacts.plot_builders import PlotTitleBuilder
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "inner_n_folds": 3,
        "seed": 42,
    }
    
    objective_computer = ObjectiveComputer(cfg)
    inner_trainer = InnerTrainer(cfg, objective_computer)
    outer_evaluator = OuterEvaluator(cfg | {"outer_eval_mode": "ensemble"})
    plot_title_builder = PlotTitleBuilder(cfg, objective_computer, "trial_test")
    
    orchestrator = OuterFoldOrchestrator(
        cfg=cfg,
        objective_computer=objective_computer,
        inner_trainer=inner_trainer,
        outer_evaluator=outer_evaluator,
        plot_title_builder=plot_title_builder,
    )
    
    assert orchestrator.cfg == cfg
    assert orchestrator.objective_computer is objective_computer
    assert orchestrator.inner_trainer is inner_trainer
    assert orchestrator.outer_evaluator is outer_evaluator


def test_outer_fold_orchestrator_run_fold_signature():
    """Test that run_fold() has the correct signature."""
    from code.training.outer_loop import OuterFoldOrchestrator
    import inspect
    
    sig = inspect.signature(OuterFoldOrchestrator.run_fold)
    params = list(sig.parameters.keys())
    
    required_params = [
        "self", "fold", "tr_idx", "te_idx", "dataset", "y_all", "groups",
        "class_names", "model_builder", "aug_transform", "input_adapter",
        "predefined_inner_splits", "optuna_trial", "global_step_offset"
    ]
    
    for param in required_params:
        assert param in params, f"Missing required parameter: {param}"


def test_outer_fold_orchestrator_result_structure():
    """Test that run_fold() returns a dictionary with all required keys."""
    from code.training.outer_loop import OuterFoldOrchestrator
    from code.training.metrics import ObjectiveComputer
    from code.training.inner_loop import InnerTrainer
    from code.training.evaluation import OuterEvaluator
    from code.artifacts.plot_builders import PlotTitleBuilder
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "inner_n_folds": 2,
        "seed": 42,
        "epochs": 1,
        "batch_size": 4,
        "lr": 0.001,
        "outer_eval_mode": "ensemble",
    }
    
    objective_computer = ObjectiveComputer(cfg)
    inner_trainer = Mock()
    outer_evaluator = Mock()
    plot_title_builder = PlotTitleBuilder(cfg, objective_computer, "trial_test")
    
    orchestrator = OuterFoldOrchestrator(
        cfg=cfg,
        objective_computer=objective_computer,
        inner_trainer=inner_trainer,
        outer_evaluator=outer_evaluator,
        plot_title_builder=plot_title_builder,
    )
    
    # Mock returns
    inner_trainer.train.return_value = {
        "best_state": {"model": "state"},
        "best_metrics": {"val_acc": 75.0, "val_macro_f1": 72.0, "val_min_per_class_f1": 65.0, "val_plur_corr": 80.0},
        "best_inner_acc": 75.0,
        "best_inner_macro_f1": 72.0,
        "best_inner_min_per_class_f1": 65.0,
        "best_inner_plur_corr": 80.0,
        "learning_curves": [],
        "tr_hist": [0.5],
        "va_hist": [0.4],
        "va_acc_hist": [75.0],
    }
    
    outer_evaluator.evaluate.return_value = {
        "y_true": [0, 1, 0, 1],
        "y_pred": [0, 1, 1, 1],
        "test_pred_rows": [],
    }
    
    # Create minimal mocks
    dataset = Mock()
    dataset.__len__ = Mock(return_value=100)
    model_builder = Mock()
    
    tr_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    te_idx = np.array([8, 9])
    y_all = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    groups = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    
    # Patch GroupKFold to avoid actual CV splitting
    with patch("code.training.outer_loop.GroupKFold") as mock_gkf:
        mock_gkf.return_value.split.return_value = [
            ([0, 1, 2, 3], [4, 5]),
            ([4, 5, 6, 7], [0, 1]),
        ]
        
        # Patch _make_loaders to avoid DataLoader creation
        with patch.object(orchestrator, "_make_loaders") as mock_loaders:
            mock_loaders.return_value = (Mock(), Mock(), Mock(), np.array([0, 1, 2, 3]))
            
            # Patch model_builder to avoid actual model creation
            mock_model = Mock()
            model_builder.return_value = mock_model
            mock_model.to.return_value = mock_model
            
            result = orchestrator.run_fold(
                fold=0,
                tr_idx=tr_idx,
                te_idx=te_idx,
                dataset=dataset,
                y_all=y_all,
                groups=groups,
                class_names=["low", "high"],
                model_builder=model_builder,
                aug_transform=None,
                input_adapter=None,
                predefined_inner_splits=None,
                optuna_trial=None,
                global_step_offset=0,
            )
    
    # Verify result structure
    required_keys = [
        "fold", "test_subjects", "y_true", "y_pred", "metrics",
        "inner_results", "best_inner_result", "fold_record",
        "learning_curves", "test_pred_rows_outer", "test_pred_rows_inner",
        "new_global_step"
    ]
    
    for key in required_keys:
        assert key in result, f"Missing required key in result: {key}"
    
    assert result["fold"] == 0
    assert isinstance(result["test_subjects"], list)
    assert isinstance(result["y_true"], list)
    assert isinstance(result["y_pred"], list)
    assert isinstance(result["metrics"], dict)


def test_outer_fold_orchestrator_inner_aggregation():
    """Test that inner results are correctly aggregated."""
    from code.training.outer_loop import OuterFoldOrchestrator
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "inner_n_folds": 3,
    }
    
    objective_computer = ObjectiveComputer(cfg)
    inner_trainer = Mock()
    outer_evaluator = Mock()
    plot_title_builder = Mock()
    
    orchestrator = OuterFoldOrchestrator(
        cfg=cfg,
        objective_computer=objective_computer,
        inner_trainer=inner_trainer,
        outer_evaluator=outer_evaluator,
        plot_title_builder=plot_title_builder,
    )
    
    # Mock inner results with varying metrics
    inner_results = [
        {
            "best_inner_acc": 70.0,
            "best_inner_macro_f1": 68.0,
            "best_inner_min_per_class_f1": 60.0,
            "best_inner_plur_corr": 75.0,
        },
        {
            "best_inner_acc": 75.0,
            "best_inner_macro_f1": 72.0,
            "best_inner_min_per_class_f1": 65.0,
            "best_inner_plur_corr": 80.0,
        },
        {
            "best_inner_acc": 80.0,
            "best_inner_macro_f1": 78.0,
            "best_inner_min_per_class_f1": 70.0,
            "best_inner_plur_corr": 85.0,
        },
    ]
    
    # Test aggregation
    mean_acc = orchestrator._aggregate_inner_metrics(inner_results, "best_inner_acc")
    mean_macro_f1 = orchestrator._aggregate_inner_metrics(inner_results, "best_inner_macro_f1")
    
    assert abs(mean_acc - 75.0) < 0.01  # (70 + 75 + 80) / 3
    assert abs(mean_macro_f1 - 72.67) < 0.1  # (68 + 72 + 78) / 3


def test_outer_fold_orchestrator_best_model_selection():
    """Test that best inner model is correctly selected based on objective."""
    from code.training.outer_loop import OuterFoldOrchestrator
    from code.training.metrics import ObjectiveComputer
    
    # Test macro F1 objective
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "inner_n_folds": 3,
    }
    
    objective_computer = ObjectiveComputer(cfg)
    orchestrator = OuterFoldOrchestrator(
        cfg=cfg,
        objective_computer=objective_computer,
        inner_trainer=Mock(),
        outer_evaluator=Mock(),
        plot_title_builder=Mock(),
    )
    
    inner_results = [
        {"best_inner_macro_f1": 68.0, "model_id": "model_0"},
        {"best_inner_macro_f1": 72.0, "model_id": "model_1"},  # Should be selected
        {"best_inner_macro_f1": 70.0, "model_id": "model_2"},
    ]
    
    best = orchestrator._select_best_inner_model(inner_results)
    assert best["model_id"] == "model_1"


def test_outer_fold_orchestrator_subject_leakage_detection():
    """Test that subject leakage is detected in inner splits."""
    from code.training.outer_loop import OuterFoldOrchestrator
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "inner_n_folds": 2,
    }
    
    objective_computer = ObjectiveComputer(cfg)
    orchestrator = OuterFoldOrchestrator(
        cfg=cfg,
        objective_computer=objective_computer,
        inner_trainer=Mock(),
        outer_evaluator=Mock(),
        plot_title_builder=Mock(),
    )
    
    # Test leakage detection
    groups = np.array([1, 1, 2, 2, 3, 3])
    inner_tr_idx = np.array([0, 1, 2, 3])  # Subjects 1, 2
    inner_va_idx = np.array([2, 3, 4, 5])  # Subjects 2, 3 - LEAKAGE!
    
    with pytest.raises(AssertionError, match="Subject leakage detected"):
        orchestrator._validate_inner_split(groups, inner_tr_idx, inner_va_idx, outer_fold=0, inner_fold=0)


def test_outer_fold_orchestrator_plots_generated():
    """Test that per-fold plots are generated (if run_dir exists)."""
    from code.training.outer_loop import OuterFoldOrchestrator
    from code.training.metrics import ObjectiveComputer
    from code.artifacts.plot_builders import PlotTitleBuilder
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
        "inner_n_folds": 2,
        "run_dir": "/tmp/test_run",
    }
    
    objective_computer = ObjectiveComputer(cfg)
    plot_title_builder = PlotTitleBuilder(cfg, objective_computer, "trial_test")
    
    orchestrator = OuterFoldOrchestrator(
        cfg=cfg,
        objective_computer=objective_computer,
        inner_trainer=Mock(),
        outer_evaluator=Mock(),
        plot_title_builder=plot_title_builder,
    )
    
    # Mock plot functions
    with patch("code.training.outer_loop.plot_confusion") as mock_plot_conf:
        with patch("code.training.outer_loop.plot_curves") as mock_plot_curves:
            
            best_inner_result = {
                "tr_hist": [0.5, 0.4],
                "va_hist": [0.4, 0.3],
                "va_acc_hist": [70.0, 75.0],
            }
            
            inner_metrics = {
                "acc": 75.0,
                "macro_f1": 72.0,
                "min_per_class_f1": 65.0,
                "plur_corr": 80.0,
            }
            
            orchestrator._generate_fold_plots(
                run_dir=Path("/tmp/test_run"),
                fold=0,
                test_subjects=[5, 8],
                y_true=[0, 1, 0, 1],
                y_pred=[0, 1, 1, 1],
                class_names=["low", "high"],
                best_inner_result=best_inner_result,
                inner_metrics=inner_metrics,
                outer_acc=75.0,
                outer_metrics={"acc": 75.0, "macro_f1": 70.0, "plur_corr": 78.0},
                per_class_f1=[0.7, 0.75],
            )
            
            # Verify plots were called
            assert mock_plot_conf.called
            assert mock_plot_curves.called


print("\n" + "=" * 80)
print("STAGE 7A TESTS: OuterFoldOrchestrator")
print("=" * 80)
print("\nThese tests will FAIL until we create code/training/outer_loop.py")
print("After extraction, all tests should PASS.\n")

