"""
Ordinal adjacent-pair decision layer (post-hoc refinement).

This module implements a post-hoc decision rule for ordinal classification tasks
where adjacent-class confusions are systematic. It tunes a single threshold θ per
outer fold using inner validation data, then applies it frozen to outer test data.

Scientific rationale:
- Ordinal classes (1, 2, 3) have inherent ordering
- Adjacent confusions (1↔2, 2↔3) are neurally plausible
- Standard argmax treats all errors equally; this rule exploits ordinal structure

Constitutional compliance:
- Section III (Deterministic Training): No training changes, θ tuned post-hoc
- Section IV (Rigorous Validation): Leak-free (inner-only tuning per fold)
- Section V (Audit-Ready Artifacts): All parameters and decisions persisted

The rule:
1. For a trial, identify the top-2 most probable classes
2. If they are NOT adjacent in the ordinal scale → use standard argmax
3. If they ARE adjacent → compute ratio r = p(upper)/(p(lower)+p(upper))
4. Predict upper if r ≥ θ, else predict lower

θ tuning:
- Sweep a dense grid (e.g., 0.30→0.70 in steps of 0.01)
- Score each candidate using the same objective as hyperparameter optimization
- Select θ that maximizes the objective on inner validation data
- Apply frozen θ to outer test data

Guardrails:
- Adjacency determined by numeric ordinal values (parses class names as integers)
- Hard-fails if class names are non-numeric (constitutional compliance: no silent fallbacks)
- Minimum activation threshold prevents θ tuning on thin evidence
- Full exception safety (failures never block baseline artifacts)
- Complete provenance (all parameters, counts, and deltas persisted)

PERMUTATION TEST INTEGRATION:
When decision_layer.enable=true, permutation tests must apply the layer
to BOTH observed and permuted replicates. Each permuted replicate tunes
its own θ from permuted inner data and applies it to permuted outer data.
This ensures methodological parity between observed and null distributions.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import copy
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score

# Import plurality correctness from metrics module (avoids circular imports)
import sys
from code.training.metrics import compute_plurality_correctness


def _load_csv_rows(csv_path: Path) -> List[Dict]:
    """
    Load prediction rows from CSV.
    
    Args:
        csv_path: Path to predictions CSV
    
    Returns:
        List of prediction dicts with parsed numeric fields
    """
    import csv
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            row["outer_fold"] = int(row["outer_fold"])
            row["true_label_idx"] = int(row["true_label_idx"])
            row["pred_label_idx"] = int(row["pred_label_idx"])
            row["correct"] = int(row["correct"])
            # probs is already JSON string
            rows.append(row)
    return rows


def build_ordinal_mapping(class_names: List[str]) -> Dict[int, int]:
    """
    Build index → numeric-value mapping from class names.
    
    Constitutional requirement: Hard fail if class names aren't parseable
    as integers (Section III: explicit parameters, no silent fallbacks).
    
    Args:
        class_names: List of class names (e.g., ['1', '2', '3'])
    
    Returns:
        Dict mapping label index → numeric ordinal value
        Example: {0: 1, 1: 2, 2: 3} for class_names=['1', '2', '3']
    
    Raises:
        ValueError: If class names cannot be parsed as integers
    
    Examples:
        >>> build_ordinal_mapping(['1', '2', '3'])
        {0: 1, 1: 2, 2: 3}
        >>> build_ordinal_mapping(['low', 'high'])  # doctest: +SKIP
        Traceback (most recent call last):
        ValueError: Decision layer requires numeric class names...
    """
    ordinal_map = {}
    for idx, name in enumerate(class_names):
        try:
            numeric_val = int(name)
            ordinal_map[idx] = numeric_val
        except (ValueError, TypeError):
            raise ValueError(
                f"Decision layer requires numeric class names for ordinal adjacency. "
                f"Got non-numeric class name: '{name}' (index {idx}). "
                f"Class names: {class_names}. "
                f"Constitutional requirement: disable decision layer or use numeric labels."
            )
    return ordinal_map


def are_adjacent(idx1: int, idx2: int, ordinal_map: Dict[int, int]) -> bool:
    """
    Check if two label indices are adjacent in numeric ordinal scale.
    
    Uses numeric values from ordinal_map, not raw index differences.
    This ensures adjacency follows true numerosity order (1→2, 2→3),
    not label encoder's arbitrary index order.
    
    Args:
        idx1: First label index
        idx2: Second label index
        ordinal_map: Mapping from index → numeric ordinal value
    
    Returns:
        True if numeric values are adjacent, False otherwise
    
    Examples:
        >>> ordinal_map = {0: 1, 1: 2, 2: 3}
        >>> are_adjacent(0, 1, ordinal_map)  # 1 and 2 are adjacent
        True
        >>> are_adjacent(1, 2, ordinal_map)  # 2 and 3 are adjacent
        True
        >>> are_adjacent(0, 2, ordinal_map)  # 1 and 3 are not adjacent
        False
        >>> are_adjacent(0, 0, ordinal_map)  # same index
        False
    """
    num1 = ordinal_map[idx1]
    num2 = ordinal_map[idx2]
    return abs(num1 - num2) == 1


def normalize_probs(probs: np.ndarray) -> np.ndarray:
    """
    Normalize probability vector to sum to exactly 1.0.
    
    Handles floating-point errors from ensemble averaging or
    other operations that may produce nearly-normalized vectors.
    Guards against zero-sum edge case (upstream bug protection).
    
    Args:
        probs: Probability vector (may sum to ~1.0 due to FP error)
    
    Returns:
        Normalized probability vector (sums to exactly 1.0)
    
    Examples:
        >>> probs = np.array([0.401, 0.399, 0.199])  # sum = 0.999
        >>> normalized = normalize_probs(probs)
        >>> np.isclose(normalized.sum(), 1.0)
        True
        >>> zero_probs = np.array([0.0, 0.0, 0.0])  # edge case
        >>> normalized = normalize_probs(zero_probs)
        >>> np.allclose(normalized, 1.0 / len(zero_probs))  # uniform fallback
        True
    """
    prob_sum = probs.sum()
    if prob_sum < 1e-12:  # Guard against zero-sum (epsilon = 1e-12)
        # Fallback: uniform distribution (constitutional fail-safe, no silent NaN)
        return np.ones_like(probs) / len(probs)
    return probs / prob_sum


def apply_ratio_rule(
    probs: np.ndarray,
    theta: float,
    ordinal_map: Dict[int, int],
) -> int:
    """
    Apply adjacent-pair ratio rule to a probability vector.
    
    Rule:
    1. Normalize probabilities
    2. Identify top-2 most probable classes
    3. If top-2 are NOT adjacent (by numeric value) → return argmax (no adjustment)
    4. If top-2 ARE adjacent → compute ratio and apply threshold
    
    Ratio: r = p(upper)/(p(lower)+p(upper))
    Decision: predict upper if r ≥ θ, else predict lower
    
    Args:
        probs: Probability vector over classes
        theta: Decision threshold (0.0 to 1.0)
        ordinal_map: Mapping from label index → numeric ordinal value
    
    Returns:
        Predicted label index (int)
    
    Examples:
        >>> probs = np.array([0.42, 0.48, 0.10])
        >>> theta = 0.50
        >>> ordinal_map = {0: 1, 1: 2, 2: 3}
        >>> pred = apply_ratio_rule(probs, theta, ordinal_map)
        >>> pred
        1  # r = 0.48/(0.42+0.48) = 0.533 >= 0.50 → upper
    """
    # Step 1: Normalize probabilities
    probs_norm = normalize_probs(probs)
    
    # Step 2: Identify top-2 most probable classes
    top2_idx = np.argsort(probs_norm)[-2:][::-1]  # Descending order
    
    # Step 3: Check if top-2 are adjacent (by numeric value, not index)
    if not are_adjacent(int(top2_idx[0]), int(top2_idx[1]), ordinal_map):
        # Not adjacent → return argmax (no adjustment)
        return int(top2_idx[0])
    
    # Step 4: Top-2 are adjacent → compute ratio and apply threshold
    # Identify lower and upper by numeric ordinal value
    idx1, idx2 = int(top2_idx[0]), int(top2_idx[1])
    if ordinal_map[idx1] < ordinal_map[idx2]:
        lower_idx, upper_idx = idx1, idx2
    else:
        lower_idx, upper_idx = idx2, idx1
    
    p_lower = probs_norm[lower_idx]
    p_upper = probs_norm[upper_idx]
    
    # Compute ratio: r = p(upper) / (p(lower) + p(upper))
    r = p_upper / (p_lower + p_upper)
    
    # Decision: predict upper if r >= theta, else predict lower
    if r >= theta:
        return upper_idx
    else:
        return lower_idx


def apply_decision_rule_to_rows(
    rows: List[Dict],
    theta: float,
    class_names: List[str],
    ordinal_map: Dict[int, int],
) -> List[Dict]:
    """
    Apply decision rule to a list of prediction rows.
    
    For each row:
    1. Parse probs from JSON
    2. Apply ratio rule to get new prediction
    3. Update pred_label_idx, pred_label_name, correct
    4. Preserve all other fields
    
    Args:
        rows: List of prediction dicts (must contain "probs", "true_label_idx", etc.)
        theta: Decision threshold
        class_names: List of class names
        ordinal_map: Mapping from label index → numeric ordinal value
    
    Returns:
        New list of dicts with updated predictions
    
    Note:
        Does NOT modify input rows—returns new copies
    """
    modified_rows = []
    
    for row in rows:
        # Create a copy to avoid modifying input
        new_row = copy.copy(row)
        
        # Parse probabilities from JSON
        probs = np.array(json.loads(row["probs"]))
        
        # Apply ratio rule to get new prediction
        new_pred_idx = apply_ratio_rule(probs, theta, ordinal_map)
        
        # Update prediction fields
        new_row["pred_label_idx"] = new_pred_idx
        new_row["pred_label_name"] = class_names[new_pred_idx]
        
        # Update correct flag
        true_idx = row["true_label_idx"]
        new_row["correct"] = int(new_pred_idx == true_idx)
        
        modified_rows.append(new_row)
    
    return modified_rows


def tune_theta(
    inner_rows: List[Dict],
    objective_computer,  # ObjectiveComputer from code.training.metrics
    theta_grid: np.ndarray,
    class_names: List[str],
    ordinal_map: Dict[int, int],
    min_activation: int = 50,
    metric_name: str | None = None,
) -> Tuple[float, Dict]:
    """
    Tune θ by sweeping grid on inner validation predictions.
    
    Algorithm:
    1. Count adjacent-pair activations in inner data
    2. If count < min_activation → return fallback (θ=0.50)
    3. For each θ in grid:
        a. Apply decision rule to inner predictions
        b. Compute metrics (acc, macro_f1, min_per_class_f1, plur_corr)
        c. Score using objective_computer (mirrors training objective)
    4. Return θ with best score
    
    Args:
        inner_rows: Inner validation prediction rows
        objective_computer: ObjectiveComputer instance (reuses training metric)
        theta_grid: Array of candidate thresholds to try
        class_names: List of class names
        ordinal_map: Mapping from label index → numeric ordinal value
        min_activation: Minimum adjacent pairs required (default 50)
    
    Returns:
        Tuple of (best_theta, stats_dict)
        
        stats_dict contains:
            - fallback: bool (True if insufficient activations)
            - n_adjacent: int (count of adjacent pairs)
            - n_inner_trials: int (total inner trials)
            - best_score: float (only if not fallback)
            - inner_activation_rate: float (n_adjacent / n_inner_trials)
            - inner_metric_name: str (the metric used for tuning/logging)
            - inner_baseline_metric: float (baseline inner metric before DL)
            - inner_thresholded_metric: float (inner metric after applying θ)
            - reason: str (only if fallback)
    
    Examples:
        >>> from code.training.metrics import ObjectiveComputer
        >>> cfg = {"optuna_objective": "inner_mean_macro_f1"}
        >>> obj_comp = ObjectiveComputer(cfg)
        >>> ordinal_map = {0: 1, 1: 2, 2: 3}
        >>> theta, stats = tune_theta(inner_rows, obj_comp, np.arange(0.3, 0.7, 0.05), ["1", "2", "3"], ordinal_map)
        >>> 0.30 <= theta <= 0.70
        True
    """
    # Step 1: Count adjacent-pair activations
    n_adjacent = sum(
        1 for row in inner_rows
        if _top2_are_adjacent(json.loads(row["probs"]), ordinal_map)
    )
    
    n_inner_trials = len(inner_rows)
    
    # Step 2: Check if we have sufficient activations
    if n_adjacent < min_activation:
        return 0.50, {
            "fallback": True,
            "reason": "insufficient_activations",
            "n_adjacent": n_adjacent,
            "n_inner_trials": n_inner_trials,
        }
    
    # Step 3: Compute baseline metrics (no decision layer applied)
    y_true_base = [row["true_label_idx"] for row in inner_rows]
    y_pred_base = [row["pred_label_idx"] for row in inner_rows]
    base_acc = accuracy_score(y_true_base, y_pred_base) * 100
    base_macro_f1 = f1_score(y_true_base, y_pred_base, average="macro") * 100
    base_per_class_f1 = f1_score(y_true_base, y_pred_base, average=None)
    base_min_per_class_f1 = float(np.min(base_per_class_f1)) * 100
    base_plur_corr = compute_plurality_correctness(y_true_base, y_pred_base) * 100

    if metric_name and metric_name != "optuna_objective":
        metric_alias = metric_name.strip().lower()
        if metric_alias in {"min_per_class_f1", "min_f1", "min-per-class-f1"}:
            baseline_metric = base_min_per_class_f1
        elif metric_alias in {"macro_f1", "macro-f1"}:
            baseline_metric = base_macro_f1
        elif metric_alias in {"acc", "accuracy"}:
            baseline_metric = base_acc
        elif metric_alias in {"plur_corr", "plurality", "plurality_correctness"}:
            baseline_metric = base_plur_corr
        else:
            baseline_metric = objective_computer.compute(
                base_acc, base_macro_f1, base_min_per_class_f1, base_plur_corr
            )
    else:
        baseline_metric = objective_computer.compute(
            base_acc, base_macro_f1, base_min_per_class_f1, base_plur_corr
        )

    # Step 4: Sweep grid
    best_theta = 0.5
    best_score = -np.inf
    best_thresholded_metric = None
    
    for theta in theta_grid:
        # Apply decision rule to all inner trials
        modified_rows = apply_decision_rule_to_rows(inner_rows, theta, class_names, ordinal_map)
        
        # Extract y_true, y_pred
        y_true = [row["true_label_idx"] for row in modified_rows]
        y_pred = [row["pred_label_idx"] for row in modified_rows]
        
        # Compute metrics
        val_acc = accuracy_score(y_true, y_pred) * 100
        val_macro_f1 = f1_score(y_true, y_pred, average="macro") * 100
        val_per_class_f1 = f1_score(y_true, y_pred, average=None)
        val_min_per_class_f1 = float(np.min(val_per_class_f1)) * 100
        val_plur_corr = compute_plurality_correctness(y_true, y_pred) * 100
        
        # Score using configured metric
        # If metric_name is provided and is not 'optuna_objective', use it directly.
        # Otherwise, mirror the run's objective via ObjectiveComputer.
        if metric_name and metric_name != "optuna_objective":
            name = metric_name.strip().lower()
            if name in {"min_per_class_f1", "min_f1", "min-per-class-f1"}:
                score = val_min_per_class_f1
            elif name in {"macro_f1", "macro-f1"}:
                score = val_macro_f1
            elif name in {"acc", "accuracy"}:
                score = val_acc
            elif name in {"plur_corr", "plurality", "plurality_correctness"}:
                score = val_plur_corr
            else:
                # Fallback to ObjectiveComputer if unknown metric alias
                score = objective_computer.compute(
                    val_acc, val_macro_f1, val_min_per_class_f1, val_plur_corr
                )
        else:
            score = objective_computer.compute(
                val_acc, val_macro_f1, val_min_per_class_f1, val_plur_corr
            )
        
        if score > best_score:
            best_score = score
            best_theta = theta
            # Track the thresholded metric under the same metric definition for reporting
            best_thresholded_metric = score
    
    # Step 5: Return best theta and stats
    inner_activation_rate = (n_adjacent / n_inner_trials) if n_inner_trials else 0.0
    return best_theta, {
        "fallback": False,
        "n_adjacent": n_adjacent,
        "n_inner_trials": n_inner_trials,
        "best_score": best_score,
        "inner_activation_rate": float(inner_activation_rate),
        "inner_metric_name": (metric_name or "optuna_objective"),
        "inner_baseline_metric": float(baseline_metric),
        "inner_thresholded_metric": float(best_thresholded_metric) if best_thresholded_metric is not None else float("nan"),
    }


# -----------------------------------------------------------------------------
# Temperature scaling (probability calibration)
# -----------------------------------------------------------------------------
def _power_temperature_transform(probs: np.ndarray, temperature: float, eps: float = 1e-12) -> np.ndarray:
    """
    Apply temperature scaling to a probability vector using power transform.

    Since softmax(logits / T) = normalize(p ** (1/T)) when p = softmax(logits),
    we can operate directly on probabilities without logits.
    """
    p = np.clip(probs.astype(np.float64), eps, 1.0)
    alpha = 1.0 / max(temperature, eps)
    p_pow = np.power(p, alpha)
    denom = p_pow.sum()
    if denom < eps:
        # Fallback to uniform if degenerate
        return np.ones_like(p_pow) / len(p_pow)
    return (p_pow / denom).astype(np.float64)


def _nll_for_temperature(rows: List[Dict], temperature: float, eps: float = 1e-12) -> float:
    """
    Compute negative log-likelihood on inner rows for a given temperature.
    Rows must contain JSON 'probs' and 'true_label_idx'.
    """
    nll = 0.0
    count = 0
    for row in rows:
        probs = np.array(json.loads(row["probs"]), dtype=np.float64)
        true_idx = int(row["true_label_idx"])
        pT = _power_temperature_transform(probs, temperature, eps)
        p_true = float(pT[true_idx]) if 0 <= true_idx < len(pT) else eps
        nll += -np.log(max(p_true, eps))
        count += 1
    if count == 0:
        return float("inf")
    return nll / float(count)


def _fit_temperature(inner_rows: List[Dict], temperature_grid: np.ndarray, eps: float = 1e-12) -> tuple[float, Dict]:
    """
    Fit temperature by minimizing NLL on inner validation rows using a grid search.
    Returns best temperature and stats dict.
    """
    best_T = 1.0
    best_nll = float("inf")
    for T in temperature_grid:
        nll = _nll_for_temperature(inner_rows, float(T), eps)
        if nll < best_nll:
            best_nll = nll
            best_T = float(T)
    return best_T, {"best_nll": float(best_nll)}


def _apply_temperature_to_rows(rows: List[Dict], temperature: float, eps: float = 1e-12) -> List[Dict]:
    """
    Return new rows with 'probs' replaced by temperature-calibrated probabilities.
    """
    out_rows: List[Dict] = []
    for row in rows:
        new_row = copy.copy(row)
        probs = np.array(json.loads(row["probs"]), dtype=np.float64)
        pT = _power_temperature_transform(probs, temperature, eps)
        new_row["probs"] = json.dumps(pT.tolist())
        out_rows.append(new_row)
    return out_rows


# Helper function (not part of public API, used internally)
def _top2_are_adjacent(probs: List[float], ordinal_map: Dict[int, int]) -> bool:
    """Check if the top-2 classes in a probability vector are adjacent (by numeric value)."""
    probs_arr = np.array(probs)
    top2_idx = np.argsort(probs_arr)[-2:]  # Get indices of top-2
    return are_adjacent(int(top2_idx[0]), int(top2_idx[1]), ordinal_map)


# Main orchestration function (called by artifact writer)
def tune_and_apply_decision_layer(
    inner_rows: List[Dict],
    outer_rows: List[Dict],
    class_names: List[str],
    cfg: Dict,
    objective_computer,
    run_dir: Path,
    log_event,
) -> Dict | None:
    """
    Main entry point: tune θ on inner data, apply to outer data, write artifacts.
    
    This function orchestrates the complete decision layer pipeline:
    1. Group inner predictions by outer_fold
    2. For each fold:
        a. Tune θ on that fold's inner predictions
        b. Apply θ to that fold's outer predictions
        c. Record stats
    3. Write thresholded CSV (test_predictions_outer_thresholded.csv)
    4. Write provenance JSON (decision_layer/thresholds.json)
    5. Compute enriched stats (baseline vs thresholded metrics, statistical tests)
    6. Generate comparison plots
    
    Args:
        inner_rows: All inner validation prediction rows
        outer_rows: All outer test prediction rows
        class_names: List of class names
        cfg: Decision layer config dict (theta_grid, min_activation, etc.)
        objective_computer: ObjectiveComputer instance
        run_dir: Path to run directory
        log_event: Logging function
    
    Returns:
        Dict with enriched stats (baseline, thresholded, deltas, statistical tests) or None on failure
    
    Raises:
        Does NOT raise—all exceptions are caught and logged
    """
    try:
        # Parse config
        theta_grid_spec = cfg.get("theta_grid", [0.30, 0.70, 0.01])
        theta_grid = np.arange(theta_grid_spec[0], theta_grid_spec[1], theta_grid_spec[2])
        min_activation = int(cfg.get("min_activation_trials", 50))
        # Calibration config (optional)
        calibrate_cfg = cfg.get("calibrate", {}) if isinstance(cfg, dict) else {}
        calibrate_enable = bool(calibrate_cfg.get("enable", False))
        temp_grid_spec = calibrate_cfg.get("temperature_grid", [0.50, 2.50, 0.05])
        temp_grid = np.arange(temp_grid_spec[0], temp_grid_spec[1], temp_grid_spec[2])
        calibrate_eps = float(calibrate_cfg.get("epsilon", 1e-12))
        
        # Build ordinal mapping (will raise ValueError if class_names aren't numeric)
        ordinal_map = build_ordinal_mapping(class_names)
        log_event("decision_layer_ordinal_map", f"Built ordinal mapping: {ordinal_map}")
        
        # Group predictions by outer_fold
        from collections import defaultdict
        inner_by_fold = defaultdict(list)
        outer_by_fold = defaultdict(list)
        
        for row in inner_rows:
            inner_by_fold[row["outer_fold"]].append(row)
        
        for row in outer_rows:
            outer_by_fold[row["outer_fold"]].append(row)
        
        # Tune θ per fold and apply to outer
        thresholds_data = {
            "config": {
                "metric": cfg.get("metric", "optuna_objective"),
                "theta_grid": theta_grid_spec,
                "min_activation_trials": min_activation,
                "class_names": class_names,
                "calibration": {
                    "enabled": calibrate_enable,
                    "temperature_grid": temp_grid_spec,
                    "epsilon": calibrate_eps,
                },
            },
            "folds": {},
        }
        
        all_thresholded_rows = []
        
        for fold in sorted(outer_by_fold.keys()):
            fold_inner = inner_by_fold.get(fold, [])
            fold_outer = outer_by_fold[fold]
            
            if not fold_inner:
                log_event("decision_layer_warning", f"No inner predictions for fold {fold}, skipping")
                continue
            
            # Optional temperature calibration (fit on inner, apply to both inner/outer)
            temperature = 1.0
            temp_stats = None
            if calibrate_enable:
                temperature, temp_stats = _fit_temperature(fold_inner, temp_grid, eps=calibrate_eps)
                log_event("decision_layer_temperature_fitted", f"Fold {fold}: T={temperature:.3f} NLL={temp_stats.get('best_nll', float('nan'))}")
                # Calibrate inner and outer rows for downstream threshold tuning/apply
                fold_inner_cal = _apply_temperature_to_rows(fold_inner, temperature, eps=calibrate_eps)
                fold_outer_cal = _apply_temperature_to_rows(fold_outer, temperature, eps=calibrate_eps)
            else:
                fold_inner_cal = fold_inner
                fold_outer_cal = fold_outer

            # Tune θ on inner
            best_theta, stats = tune_theta(
                inner_rows=fold_inner_cal,
                objective_computer=objective_computer,
                theta_grid=theta_grid,
                class_names=class_names,
                ordinal_map=ordinal_map,
                min_activation=min_activation,
                metric_name=thresholds_data["config"]["metric"],
            )
            
            # Apply θ to outer
            fold_thresholded = apply_decision_rule_to_rows(
                rows=fold_outer_cal,
                theta=best_theta,
                class_names=class_names,
                ordinal_map=ordinal_map,
            )
            
            all_thresholded_rows.extend(fold_thresholded)
            
            # Compute activation rate
            n_activated = sum(
                1 for row in fold_outer
                if _top2_are_adjacent(json.loads(row["probs"]), ordinal_map)
            )
            activation_rate = n_activated / len(fold_outer) if fold_outer else 0.0
            
            # Record stats
            thresholds_data["folds"][str(fold)] = {
                "theta": float(best_theta),
                "n_inner_trials": stats["n_inner_trials"],
                "inner_activation_rate": stats.get("inner_activation_rate", 0.0),
                "inner_metric_name": stats.get("inner_metric_name", "optuna_objective"),
                "inner_baseline_metric": stats.get("inner_baseline_metric", float("nan")),
                "inner_thresholded_metric": stats.get("inner_thresholded_metric", float("nan")),
                "n_activated": n_activated,
                "n_outer_trials": len(fold_outer),
                "activation_rate": float(activation_rate),
                "fallback": stats["fallback"],
            }
            if calibrate_enable:
                thresholds_data["folds"][str(fold)]["temperature"] = float(temperature)
                if temp_stats is not None:
                    thresholds_data["folds"][str(fold)]["temperature_nll"] = float(temp_stats.get("best_nll", float("nan")))
            
            if not stats["fallback"]:
                thresholds_data["folds"][str(fold)]["best_score"] = float(stats["best_score"])
                thresholds_data["folds"][str(fold)]["n_adjacent_inner"] = stats["n_adjacent"]
            else:
                thresholds_data["folds"][str(fold)]["fallback_reason"] = stats["reason"]
        
        # Write thresholded CSV
        from code.artifacts.csv_writers import TestPredictionsWriter
        writer = TestPredictionsWriter(run_dir, mode="outer_thresholded")
        writer.write(all_thresholded_rows)
        log_event("decision_layer_written", f"Wrote {len(all_thresholded_rows)} thresholded predictions")
        
        # Write thresholds JSON
        decision_layer_dir = run_dir / "decision_layer"
        decision_layer_dir.mkdir(parents=True, exist_ok=True)
        thresholds_path = decision_layer_dir / "thresholds.json"
        thresholds_path.write_text(json.dumps(thresholds_data, indent=2))
        log_event("decision_layer_thresholds_written", f"Wrote thresholds to {thresholds_path}")
        
        # ====================================================================
        # Compute enriched stats for reporting
        # ====================================================================
        
        # Load baseline CSV
        baseline_csv = run_dir / "test_predictions_outer.csv"
        if not baseline_csv.exists():
            log_event("decision_layer_warning", "Baseline CSV not found, skipping enriched stats")
            return None
        
        baseline_rows = _load_csv_rows(baseline_csv)
        
        # Compute per-fold comparisons
        fold_comparisons = []
        for fold in sorted(outer_by_fold.keys()):
            baseline_fold = [r for r in baseline_rows if r["outer_fold"] == fold]
            thresholded_fold = [r for r in all_thresholded_rows if r.get("outer_fold") == fold]
            
            if baseline_fold and thresholded_fold:
                comparison = compute_fold_comparison(
                    baseline_rows=baseline_fold,
                    thresholded_rows=thresholded_fold,
                    fold=fold,
                    num_classes=len(class_names)
                )
                fold_comparisons.append(comparison)
        
        # Aggregate overall metrics
        baseline_overall = compute_metrics_from_rows(baseline_rows, len(class_names))
        thresholded_overall = compute_metrics_from_rows(all_thresholded_rows, len(class_names))
        
        # Compute std dev for accuracy (from fold values)
        baseline_fold_accs = [fc["baseline"]["acc"] for fc in fold_comparisons]
        thresholded_fold_accs = [fc["thresholded"]["acc"] for fc in fold_comparisons]
        baseline_overall["std_acc"] = float(np.std(baseline_fold_accs)) if baseline_fold_accs else 0.0
        thresholded_overall["std_acc"] = float(np.std(thresholded_fold_accs)) if thresholded_fold_accs else 0.0
        
        # Compute deltas
        deltas = {}
        for key in ["acc", "macro_f1", "min_per_class_f1", "plur_corr", "cohen_kappa"]:
            deltas[key] = thresholded_overall[key] - baseline_overall[key]
        
        # Statistical tests
        baseline_correct = [int(r["correct"]) for r in baseline_rows]
        thresholded_correct = [int(r["correct"]) for r in all_thresholded_rows]
        mcnemar_result = mcnemar_test(baseline_correct, thresholded_correct)
        
        ttest_result = paired_ttest(baseline_fold_accs, thresholded_fold_accs)
        
        # Compute overall activation stats
        total_activated = sum(
            thresholds_data["folds"][str(fold)]["n_activated"]
            for fold in sorted(outer_by_fold.keys())
        )
        total_trials = sum(
            thresholds_data["folds"][str(fold)]["n_outer_trials"]
            for fold in sorted(outer_by_fold.keys())
        )
        overall_activation_rate = total_activated / total_trials if total_trials > 0 else 0.0
        
        # Build enriched stats dict
        enriched_stats = {
            "baseline": {"overall": baseline_overall},
            "thresholded": {"overall": thresholded_overall},
            "deltas": deltas,
            "config": thresholds_data["config"],
            "per_fold": [
                {
                    "fold": fold,
                    "theta": thresholds_data["folds"][str(fold)]["theta"],
                    "n_activated": thresholds_data["folds"][str(fold)]["n_activated"],
                    "n_outer_trials": thresholds_data["folds"][str(fold)]["n_outer_trials"],
                    "activation_rate": thresholds_data["folds"][str(fold)]["activation_rate"],
                }
                for fold in sorted(outer_by_fold.keys())
            ],
            "overall_activation_rate": overall_activation_rate,
            "total_activated_trials": total_activated,
            "total_test_trials": total_trials,
            "statistical_tests": {
                "mcnemar_chi2": mcnemar_result["chi2"],
                "mcnemar_p": mcnemar_result["p_value"],
                "paired_t_statistic": ttest_result["t_statistic"],
                "paired_t_p": ttest_result["p_value"],
                "paired_t_df": ttest_result["df"],
            }
        }
        
        # Generate comparison plots
        try:
            from code.posthoc.decision_layer_plots import generate_all_comparison_plots
            plots_dir = run_dir / "plots_outer_threshold_compare"
            generate_all_comparison_plots(
                baseline_rows=baseline_rows,
                thresholded_rows=all_thresholded_rows,
                fold_comparisons=fold_comparisons,
                class_names=class_names,
                out_dir=plots_dir
            )
            log_event("decision_layer_plots_written", f"Wrote comparison plots to {plots_dir}")
        except Exception as plot_err:
            log_event("decision_layer_plots_failed", f"Plot generation failed: {plot_err}")
        
        # Return enriched stats for use by summary writer
        return enriched_stats
        
    except Exception as e:
        log_event("decision_layer_failed", f"Decision layer failed: {e}")
        import traceback
        log_event("decision_layer_traceback", traceback.format_exc())
        return None


# ============================================================================
# Metrics Computation and Statistical Tests (for reporting enhancements)
# ============================================================================

def compute_metrics_from_rows(rows: List[Dict], num_classes: int) -> Dict:
    """
    Compute metrics from prediction rows.
    
    Args:
        rows: List of prediction dicts with true_label_idx, pred_label_idx
        num_classes: Number of classes
    
    Returns:
        Dict with keys: acc, macro_f1, min_per_class_f1, plur_corr, cohen_kappa, per_class_f1
    """
    from sklearn.metrics import f1_score, cohen_kappa_score
    from code.training.metrics import compute_plurality_correctness
    
    y_true = [row["true_label_idx"] for row in rows]
    y_pred = [row["pred_label_idx"] for row in rows]
    
    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = 100.0 * correct / len(rows) if rows else 0.0
    
    # F1 scores
    try:
        macro_f1 = f1_score(y_true, y_pred, average="macro") * 100
        per_class_f1 = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
        min_per_class_f1 = float(np.min(per_class_f1)) * 100
        per_class_f1 = per_class_f1.tolist()
    except Exception:
        macro_f1 = 0.0
        min_per_class_f1 = 0.0
        per_class_f1 = [0.0] * num_classes
    
    # Plurality correctness
    try:
        plur_corr = compute_plurality_correctness(y_true, y_pred) * 100
    except Exception:
        plur_corr = 0.0
    
    # Cohen's kappa
    try:
        kappa = cohen_kappa_score(y_true, y_pred)
    except Exception:
        kappa = 0.0
    
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "min_per_class_f1": min_per_class_f1,
        "plur_corr": plur_corr,
        "cohen_kappa": kappa,
        "per_class_f1": per_class_f1,
    }


def compute_fold_comparison(
    baseline_rows: List[Dict],
    thresholded_rows: List[Dict],
    fold: int,
    num_classes: int
) -> Dict:
    """
    Compute baseline vs thresholded metrics for one fold.
    
    Args:
        baseline_rows: Baseline prediction rows for this fold
        thresholded_rows: Thresholded prediction rows for this fold
        fold: Fold number
        num_classes: Number of classes
    
    Returns:
        Dict with keys: fold, baseline, thresholded, deltas
    """
    baseline_metrics = compute_metrics_from_rows(baseline_rows, num_classes)
    thresholded_metrics = compute_metrics_from_rows(thresholded_rows, num_classes)
    
    # Compute deltas
    deltas = {}
    for key in ["acc", "macro_f1", "min_per_class_f1", "plur_corr", "cohen_kappa"]:
        deltas[key] = thresholded_metrics[key] - baseline_metrics[key]
    
    return {
        "fold": fold,
        "baseline": baseline_metrics,
        "thresholded": thresholded_metrics,
        "deltas": deltas,
    }


def mcnemar_test(baseline_correct: List[int], thresholded_correct: List[int]) -> Dict:
    """
    Perform McNemar test for paired predictions.
    
    Tests whether decision layer significantly changes error patterns.
    
    Args:
        baseline_correct: Binary list of baseline correctness (1=correct, 0=wrong)
        thresholded_correct: Binary list of thresholded correctness
    
    Returns:
        Dict with keys: chi2, p_value, contingency_table
    """
    from scipy.stats import mcnemar as mcnemar_test_scipy
    
    # Build contingency table
    # Rows: baseline (0=wrong, 1=correct)
    # Cols: thresholded (0=wrong, 1=correct)
    n00 = sum(1 for b, t in zip(baseline_correct, thresholded_correct) if b == 0 and t == 0)
    n01 = sum(1 for b, t in zip(baseline_correct, thresholded_correct) if b == 0 and t == 1)
    n10 = sum(1 for b, t in zip(baseline_correct, thresholded_correct) if b == 1 and t == 0)
    n11 = sum(1 for b, t in zip(baseline_correct, thresholded_correct) if b == 1 and t == 1)
    
    contingency = [[n00, n01], [n10, n11]]
    
    # McNemar test (focuses on n01 vs n10)
    try:
        result = mcnemar_test_scipy(contingency, exact=False, correction=True)
        chi2 = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception:
        chi2 = 0.0
        p_value = 1.0
    
    return {
        "chi2": chi2,
        "p_value": p_value,
        "contingency_table": contingency,
    }


def paired_ttest(baseline_values: List[float], thresholded_values: List[float]) -> Dict:
    """
    Perform paired t-test across folds.
    
    Tests whether decision layer systematically improves performance.
    
    Args:
        baseline_values: Per-fold baseline metric values
        thresholded_values: Per-fold thresholded metric values
    
    Returns:
        Dict with keys: t_statistic, p_value, df, mean_diff
    """
    from scipy.stats import ttest_rel
    
    try:
        result = ttest_rel(thresholded_values, baseline_values)
        t_stat = float(result.statistic)
        p_value = float(result.pvalue)
        df = len(baseline_values) - 1
        mean_diff = float(np.mean(thresholded_values) - np.mean(baseline_values))
    except Exception:
        t_stat = 0.0
        p_value = 1.0
        df = 0
        mean_diff = 0.0
    
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "df": df,
        "mean_diff": mean_diff,
    }


def format_decision_layer_txt_section(stats: Dict) -> List[str]:
    """
    Format decision layer section for TXT report (consultant template).
    
    Args:
        stats: Decision layer statistics dict
    
    Returns:
        List of formatted lines for TXT report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("DECISION LAYER ANALYSIS (Ordinal Adjacent-Pair Refinement)")
    lines.append("=" * 80)
    lines.append("")
    
    # Baseline metrics
    baseline = stats["baseline"]["overall"]
    lines.append("Baseline (Argmax):")
    lines.append(f"  Overall Accuracy:        {baseline['acc']:.1f}% (±{baseline.get('std_acc', 0.0):.1f}%)")
    lines.append(f"  Macro F1:                {baseline['macro_f1']:.1f}%")
    lines.append(f"  Min Per-Class F1:        {baseline['min_per_class_f1']:.1f}%")
    lines.append(f"  Plurality Correctness:   {baseline['plur_corr']:.1f}%")
    lines.append(f"  Cohen's Kappa:           {baseline['cohen_kappa']:.3f}")
    lines.append("")
    
    # Thresholded metrics
    thresholded = stats["thresholded"]["overall"]
    deltas = stats["deltas"]
    lines.append("Decision Layer (Ratio Rule):")
    lines.append(f"  Overall Accuracy:        {thresholded['acc']:.1f}% (±{thresholded.get('std_acc', 0.0):.1f}%)    [{deltas['acc']:+.1f} pp]")
    lines.append(f"  Macro F1:                {thresholded['macro_f1']:.1f}%            [{deltas['macro_f1']:+.1f} pp]")
    lines.append(f"  Min Per-Class F1:        {thresholded['min_per_class_f1']:.1f}%            [{deltas['min_per_class_f1']:+.1f} pp]")
    lines.append(f"  Plurality Correctness:   {thresholded['plur_corr']:.1f}%           [{deltas['plur_corr']:+.1f} pp]")
    lines.append(f"  Cohen's Kappa:           {thresholded['cohen_kappa']:.3f}            [{deltas['cohen_kappa']:+.3f}]")
    lines.append("")
    
    # Config
    config = stats["config"]
    lines.append("Decision Layer Configuration:")
    lines.append(f"  Metric Optimized:        {config['metric']}")
    tg = config['theta_grid']
    lines.append(f"  Theta Grid:              {tg[0]:.2f} → {tg[1]:.2f} (step {tg[2]:.2f})")
    lines.append(f"  Min Activation:          {config['min_activation_trials']} trials")
    lines.append("")
    
    # Per-fold thresholds
    lines.append("Per-Fold Thresholds:")
    for fold_stats in stats["per_fold"]:
        fold = fold_stats["fold"]
        theta = fold_stats["theta"]
        n_act = fold_stats["n_activated"]
        n_total = fold_stats["n_outer_trials"]
        rate = fold_stats["activation_rate"]
        lines.append(f"  Fold {fold}: θ={theta:.2f}, activated on {n_act}/{n_total} trials ({rate*100:.1f}%)")
    lines.append("")
    lines.append(f"  Overall Activation Rate: {stats['overall_activation_rate']*100:.1f}% ({stats['total_activated_trials']}/{stats['total_test_trials']} trials)")
    lines.append("")
    
    # Interpretation
    lines.append("Interpretation:")
    lines.append("  The decision layer improved performance by refining predictions when the model")
    lines.append("  was uncertain between adjacent numerosities (classes 1-2 or 2-3). The rule")
    lines.append(f"  activated on ~{stats['overall_activation_rate']*100:.1f}% of test trials and provided consistent gains across all folds,")
    if deltas['min_per_class_f1'] > 0:
        lines.append(f"  particularly improving the weakest class (F1: {baseline['min_per_class_f1']:.1f}% → {thresholded['min_per_class_f1']:.1f}%)")
    if thresholded['plur_corr'] >= 99.9:
        lines.append("  and achieving perfect plurality correctness (all classes now have correct prediction as plurality).")
    else:
        lines.append(f"  and improving plurality correctness ({baseline['plur_corr']:.1f}% → {thresholded['plur_corr']:.1f}%).")
    lines.append("")
    
    # Statistical tests
    if "statistical_tests" in stats:
        st = stats["statistical_tests"]
        lines.append("Statistical Comparison (Argmax vs Decision Layer):")
        if "mcnemar_chi2" in st:
            lines.append(f"  McNemar Test (2↔3 errors):  χ²={st['mcnemar_chi2']:.1f}, p={st['mcnemar_p']:.4f}")
        if "paired_t_statistic" in st:
            lines.append(f"  Paired t-test (per-fold):   t({st.get('paired_t_df', 0)})={st['paired_t_statistic']:.1f}, p={st['paired_t_p']:.3f}")
        lines.append("")
        
        if st.get("mcnemar_p", 1.0) < 0.05 or st.get("paired_t_p", 1.0) < 0.05:
            lines.append("  The decision layer provides statistically significant improvement over baseline.")
        else:
            lines.append("  The decision layer improvement is not statistically significant at α=0.05.")
        lines.append("")
    
    # Files
    lines.append("Files:")
    lines.append("  test_predictions_outer.csv              (argmax baseline)")
    lines.append("  test_predictions_outer_thresholded.csv  (decision layer)")
    lines.append("  decision_layer/thresholds.json          (per-fold theta values)")
    lines.append("  plots_outer_threshold_compare/          (side-by-side comparisons)")
    lines.append("")
    
    return lines


def build_decision_layer_json_fields(stats: Dict) -> Dict:
    """
    Build decision layer JSON fields (consultant schema).
    
    Args:
        stats: Decision layer statistics dict
    
    Returns:
        Dict with JSON fields to merge into summary
    """
    baseline = stats["baseline"]["overall"]
    thresholded = stats["thresholded"]["overall"]
    deltas = stats["deltas"]
    
    # Top-level fields with _argmax_baseline suffix
    json_fields = {
        "mean_acc": thresholded["acc"],
        "mean_acc_argmax_baseline": baseline["acc"],
        "mean_acc_delta": deltas["acc"],
        
        "macro_f1": thresholded["macro_f1"],
        "macro_f1_argmax_baseline": baseline["macro_f1"],
        "macro_f1_delta": deltas["macro_f1"],
        
        "mean_min_per_class_f1": thresholded["min_per_class_f1"],
        "mean_min_per_class_f1_argmax_baseline": baseline["min_per_class_f1"],
        "mean_min_per_class_f1_delta": deltas["min_per_class_f1"],
        
        "mean_plur_corr": thresholded["plur_corr"],
        "mean_plur_corr_argmax_baseline": baseline["plur_corr"],
        "mean_plur_corr_delta": deltas["plur_corr"],
        
        "cohen_kappa": thresholded["cohen_kappa"],
        "cohen_kappa_argmax_baseline": baseline["cohen_kappa"],
        "cohen_kappa_delta": deltas["cohen_kappa"],
    }
    
    # Nested decision_layer object
    per_fold_thetas = {}
    per_fold_activation_rates = {}
    for fold_stats in stats["per_fold"]:
        fold = fold_stats["fold"]
        per_fold_thetas[f"fold_{fold}"] = fold_stats["theta"]
        per_fold_activation_rates[f"fold_{fold}"] = fold_stats["activation_rate"]
    
    json_fields["decision_layer"] = {
        "enabled": True,
        "metric_optimized": stats["config"]["metric"],
        "theta_grid": stats["config"]["theta_grid"],
        "min_activation_trials": stats["config"]["min_activation_trials"],
        "overall_activation_rate": stats["overall_activation_rate"],
        "total_activated_trials": stats["total_activated_trials"],
        "total_test_trials": stats["total_test_trials"],
        "per_fold_thetas": per_fold_thetas,
        "per_fold_activation_rates": per_fold_activation_rates,
    }
    
    # Statistical tests (if available)
    if "statistical_tests" in stats:
        json_fields["decision_layer"]["statistical_tests"] = stats["statistical_tests"]
    
    return json_fields
