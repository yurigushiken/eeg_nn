"""
Tests for PlotTitleBuilder class (artifacts/plot_builders.py).

These tests verify that plot title generation works correctly:
- Objective-specific metric labels
- Fold titles (simple and enhanced)
- Overall titles
- Per-class F1 info formatting

Constitutional compliance: Section V (Audit-Ready Artifacts)
"""
import pytest


def test_plot_builders_imports():
    """Test that PlotTitleBuilder can be imported (will fail pre-extraction)."""
    try:
        from code.artifacts.plot_builders import PlotTitleBuilder
        assert PlotTitleBuilder is not None
    except ImportError as e:
        pytest.fail(f"Cannot import PlotTitleBuilder: {e}")


def test_plot_title_builder_init():
    """Test PlotTitleBuilder initialization."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "inner_mean_macro_f1",
    }
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "trial_12345")
    
    assert builder.trial_dir_name == "trial_12345"
    assert builder.obj_computer is obj_computer


def test_build_objective_label_macro_f1():
    """Test objective label for macro F1."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_macro_f1"}
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "test_trial")
    
    inner_metrics = {
        "acc": 75.5,
        "macro_f1": 72.3,
        "min_per_class_f1": 65.0,
        "plur_corr": 80.0,
    }
    
    label = builder.build_objective_label(inner_metrics)
    assert "inner-mean macro-F1=72.30" in label


def test_build_objective_label_min_per_class_f1():
    """Test objective label for min per-class F1."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_min_per_class_f1"}
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "test_trial")
    
    inner_metrics = {
        "acc": 75.5,
        "macro_f1": 72.3,
        "min_per_class_f1": 65.0,
        "plur_corr": 80.0,
    }
    
    label = builder.build_objective_label(inner_metrics)
    assert "inner-mean min-per-class-F1=65.00" in label


def test_build_objective_label_composite_threshold():
    """Test objective label for composite (threshold mode)."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "min_f1_threshold": 40.0,
    }
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "test_trial")
    
    # Below threshold
    inner_metrics = {
        "acc": 75.5,
        "macro_f1": 72.3,
        "min_per_class_f1": 35.0,  # Below 40
        "plur_corr": 80.0,
    }
    
    label = builder.build_objective_label(inner_metrics)
    assert "composite=3.50" in label  # 35.0 * 0.1
    assert "gradient" in label
    assert "threshold=40.0" in label


def test_build_objective_label_composite_weighted():
    """Test objective label for composite (weighted mode)."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {
        "optuna_objective": "composite_min_f1_plur_corr",
        "composite_min_f1_weight": 0.6,
    }
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "test_trial")
    
    inner_metrics = {
        "acc": 75.5,
        "macro_f1": 72.3,
        "min_per_class_f1": 60.0,
        "plur_corr": 70.0,
    }
    
    label = builder.build_objective_label(inner_metrics)
    # composite = 0.6 * 60 + 0.4 * 70 = 36 + 28 = 64
    assert "composite=64.00" in label
    assert "weight=0.60" in label


def test_build_fold_title_simple():
    """Test simple fold title (for basic plots)."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_macro_f1"}
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "trial_abc")
    
    inner_metrics = {"acc": 75.0, "macro_f1": 72.0, "min_per_class_f1": 65.0, "plur_corr": 80.0}
    outer_metrics = {"acc": 68.0, "macro_f1": 65.0, "plur_corr": 75.0}
    
    title = builder.build_fold_title_simple(
        fold=0,
        test_subjects=[2, 5, 8],
        inner_metrics=inner_metrics,
        outer_acc=68.0,
    )
    
    assert "trial_abc" in title
    assert "Fold 1" in title
    assert "[2, 5, 8]" in title
    assert "inner-mean macro-F1=72.00" in title
    assert "acc=68.00" in title


def test_build_fold_title_enhanced():
    """Test enhanced fold title (with inner/outer comparison)."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_macro_f1"}
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "trial_xyz")
    
    inner_metrics = {"acc": 75.0, "macro_f1": 72.0, "min_per_class_f1": 65.0, "plur_corr": 80.0}
    outer_metrics = {"acc": 68.0, "macro_f1": 65.0, "plur_corr": 75.0}
    per_class_f1 = [0.70, 0.65, 0.60]
    
    title = builder.build_fold_title_enhanced(
        fold=0,
        test_subjects=[2, 5],
        inner_metrics=inner_metrics,
        outer_metrics=outer_metrics,
        per_class_f1=per_class_f1,
    )
    
    assert "trial_xyz" in title
    assert "Fold 1" in title
    assert "inner" in title
    assert "outer" in title
    assert "acc=68.00" in title


def test_build_overall_title_simple():
    """Test simple overall title."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_macro_f1"}
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "trial_overall")
    
    inner_metrics = {"acc": 75.0, "macro_f1": 72.0, "min_per_class_f1": 65.0, "plur_corr": 80.0}
    
    title = builder.build_overall_title_simple(
        inner_metrics=inner_metrics,
        outer_mean_acc=68.5,
    )
    
    assert "trial_overall" in title
    assert "Overall" in title
    assert "inner-mean macro-F1=72.00" in title
    assert "mean_acc=68.50" in title


def test_build_overall_title_enhanced():
    """Test enhanced overall title (with inner/outer comparison)."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_min_per_class_f1"}
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "trial_final")
    
    inner_metrics = {"acc": 75.0, "macro_f1": 72.0, "min_per_class_f1": 65.0, "plur_corr": 80.0}
    outer_metrics = {"acc": 68.0, "macro_f1": 65.0, "min_per_class_f1": 60.0, "plur_corr": 75.0}
    
    title = builder.build_overall_title_enhanced(
        inner_metrics=inner_metrics,
        outer_metrics=outer_metrics,
        outer_mean_acc=68.0,
    )
    
    assert "trial_final" in title
    assert "Overall" in title
    assert "inner" in title
    assert "outer" in title


def test_build_per_class_info():
    """Test per-class F1 info formatting."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_macro_f1"}
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "test")
    
    per_class_f1 = [0.75, 0.68, 0.82]
    class_names = ["low", "medium", "high"]
    
    lines = builder.build_per_class_info(per_class_f1, class_names)
    
    assert len(lines) == 4  # Header + 3 classes
    assert "Per-class F1" in lines[0]
    assert "low: 75.00%" in lines[1]
    assert "medium: 68.00%" in lines[2]
    assert "high: 82.00%" in lines[3]


def test_build_per_class_info_empty():
    """Test per-class F1 info with empty data."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    cfg = {"optuna_objective": "inner_mean_macro_f1"}
    obj_computer = ObjectiveComputer(cfg)
    builder = PlotTitleBuilder(cfg, obj_computer, "test")
    
    lines = builder.build_per_class_info(None, ["a", "b"])
    assert len(lines) == 0
    
    lines = builder.build_per_class_info([], ["a", "b"])
    assert len(lines) == 0


def test_all_objectives_supported():
    """Test that all objectives produce valid labels."""
    from code.artifacts.plot_builders import PlotTitleBuilder
    from code.training.metrics import ObjectiveComputer
    
    objectives = [
        "inner_mean_macro_f1",
        "inner_mean_min_per_class_f1",
        "inner_mean_plur_corr",
        "inner_mean_acc",
    ]
    
    inner_metrics = {
        "acc": 75.0,
        "macro_f1": 72.0,
        "min_per_class_f1": 65.0,
        "plur_corr": 80.0,
    }
    
    for obj in objectives:
        cfg = {"optuna_objective": obj}
        obj_computer = ObjectiveComputer(cfg)
        builder = PlotTitleBuilder(cfg, obj_computer, "test")
        
        label = builder.build_objective_label(inner_metrics)
        assert len(label) > 0
        assert "inner-mean" in label
        
        print(f"  [OK] {obj}: {label}")


print("\n" + "=" * 80)
print("STAGE 3 TESTS: PlotTitleBuilder")
print("=" * 80)
print("\nThese tests will FAIL until we extract PlotTitleBuilder.")
print("After extraction, all tests should PASS.\n")

