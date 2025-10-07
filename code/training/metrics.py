"""
Objective-aligned metric computation for model selection and checkpointing.

This module encapsulates the logic for computing objective metrics that align
with Optuna optimization objectives. It ensures that pruning, checkpoint selection,
and early stopping all use the same metric that Optuna is optimizing for.

Constitutional compliance: Section III (Deterministic Training)
- Explicit parameter specification (fail-fast on missing params)
- No silent fallbacks for scientifically critical parameters

Classes:
    ObjectiveComputer: Computes objective-aligned metrics for all supported objectives
"""

from __future__ import annotations
from typing import Dict


class ObjectiveComputer:
    """
    Encapsulates all objective-aligned metric computation.
    
    Supports 5 optimization objectives:
    1. inner_mean_macro_f1: Maximize validation macro F1-score
    2. inner_mean_min_per_class_f1: Maximize minimum per-class F1-score (fairness)
    3. inner_mean_acc: Maximize validation accuracy
    4. inner_mean_plur_corr: Maximize plurality correctness (distinctness)
    5. composite_min_f1_plur_corr: Dual-objective with two modes:
       - Threshold mode: Meet decodability threshold, then maximize distinctness
       - Weighted mode: Linear combination of decodability and distinctness
    
    Per constitution Section III: Critical parameters must be explicitly specified.
    No silent defaults allowed for scientifically important parameters.
    
    Example usage:
        >>> cfg = {
        ...     "optuna_objective": "composite_min_f1_plur_corr",
        ...     "min_f1_threshold": 35.0
        ... }
        >>> computer = ObjectiveComputer(cfg)
        >>> metric = computer.compute(
        ...     val_acc=50.0,
        ...     val_macro_f1=45.0,
        ...     val_min_per_class_f1=36.0,
        ...     val_plur_corr=85.0
        ... )
        >>> # Returns 85.0 (plur_corr) because min_f1 >= threshold
    """
    
    def __init__(self, cfg: Dict):
        """
        Initialize ObjectiveComputer with configuration.
        
        Args:
            cfg: Configuration dictionary containing:
                - optuna_objective: One of the 5 supported objectives
                - For composite_min_f1_plur_corr only:
                    - min_f1_threshold: float [0, 100] (threshold mode), OR
                    - composite_min_f1_weight: float [0, 1] (weighted mode)
        
        Raises:
            ValueError: If objective is invalid or required parameters missing
        """
        self.objective = cfg.get("optuna_objective", "")
        if not self.objective:
            raise ValueError(
                "'optuna_objective' must be explicitly specified in config for scientific validity. "
                "No fallback allowed. Choose from: inner_mean_macro_f1, inner_mean_min_per_class_f1, "
                "inner_mean_plur_corr, inner_mean_acc, composite_min_f1_plur_corr"
            )
        
        # Parse composite parameters if composite objective
        if self.objective == "composite_min_f1_plur_corr":
            self.params = self._parse_composite_params(cfg)
        else:
            self.params = {"mode": "direct"}
    
    def _parse_composite_params(self, cfg: Dict) -> Dict:
        """
        Parse and validate composite objective parameters with constitutional fail-fast.
        
        Supports two modes:
        1. Threshold mode: min_f1_threshold specified (recommended for "distinct + decodable")
        2. Weighted mode: composite_min_f1_weight specified (backward compatible)
        
        Args:
            cfg: Configuration dictionary
        
        Returns:
            dict: {"mode": "threshold", "threshold": float} or {"mode": "weighted", "weight": float}
        
        Raises:
            ValueError: If neither, both, or invalid parameters specified
        """
        has_threshold = "min_f1_threshold" in cfg
        has_weight = "composite_min_f1_weight" in cfg
        
        if has_threshold and has_weight:
            raise ValueError(
                "Ambiguous config: both min_f1_threshold and composite_min_f1_weight specified. "
                "Use only ONE: min_f1_threshold (threshold approach) OR composite_min_f1_weight (weighted approach)"
            )
        
        if not has_threshold and not has_weight:
            raise ValueError(
                "composite_min_f1_plur_corr objective requires either:\n"
                "  - min_f1_threshold: 35.0  (threshold approach - recommended)\n"
                "  - composite_min_f1_weight: 0.50  (weighted approach - backward compatible)"
            )
        
        if has_threshold:
            threshold = float(cfg["min_f1_threshold"])
            if not (0.0 <= threshold <= 100.0):
                raise ValueError(f"min_f1_threshold must be in range [0.0, 100.0], got {threshold}")
            return {"mode": "threshold", "threshold": threshold}
        else:
            weight = float(cfg["composite_min_f1_weight"])
            if not (0.0 <= weight <= 1.0):
                raise ValueError(f"composite_min_f1_weight must be in range [0.0, 1.0], got {weight}")
            return {"mode": "weighted", "weight": weight}
    
    def compute(
        self,
        val_acc: float,
        val_macro_f1: float,
        val_min_per_class_f1: float,
        val_plur_corr: float,
    ) -> float:
        """
        Compute the per-epoch metric aligned with the configured Optuna objective.
        
        This ensures that pruning and checkpoint selection use the same metric
        that Optuna is optimizing for, maintaining scientific integrity.
        
        Args:
            val_acc: Validation accuracy (0-100)
            val_macro_f1: Validation macro-F1 (0-100)
            val_min_per_class_f1: Validation min per-class F1 (0-100)
            val_plur_corr: Validation plurality correctness (0-100)
        
        Returns:
            float: The metric value aligned with the optimization objective (to be maximized)
        
        Raises:
            ValueError: If objective is invalid
        
        Example:
            >>> computer = ObjectiveComputer({"optuna_objective": "inner_mean_macro_f1"})
            >>> metric = computer.compute(50.0, 45.0, 30.0, 80.0)
            >>> assert metric == 45.0  # Returns macro_f1
        """
        if self.objective == "inner_mean_min_per_class_f1":
            return val_min_per_class_f1
        elif self.objective == "inner_mean_acc":
            return val_acc
        elif self.objective == "inner_mean_plur_corr":
            return val_plur_corr
        elif self.objective == "composite_min_f1_plur_corr":
            # Dual-mode support: threshold (recommended) or weighted (backward compatible)
            if self.params["mode"] == "threshold":
                # Threshold approach with gradient below threshold
                # Gradient is critical for:
                # 1. Optuna pruning: distinguishes "bad" (20%) from "close" (37%)
                # 2. TPE sampling: learns which hyperparameters approach threshold
                threshold = self.params["threshold"]
                if val_min_per_class_f1 < threshold:
                    # Small gradient proportional to distance from 0
                    # Range: 0.0 - 3.8 (for threshold=38%)
                    return val_min_per_class_f1 * 0.1
                else:
                    # Above threshold: maximize distinctness
                    # Range: 0.0 - 100.0 (plur_corr percentage)
                    return val_plur_corr
            else:  # weighted mode
                weight = self.params["weight"]
                return weight * val_min_per_class_f1 + (1.0 - weight) * val_plur_corr
        elif self.objective == "inner_mean_macro_f1":
            return val_macro_f1
        else:
            raise ValueError(
                f"Invalid objective in compute: '{self.objective}'. "
                f"Must be one of: inner_mean_macro_f1, inner_mean_min_per_class_f1, "
                f"inner_mean_plur_corr, inner_mean_acc, composite_min_f1_plur_corr"
            )
    
    def get_mode(self) -> str:
        """
        Return the objective mode for display/logging purposes.
        
        Returns:
            str: One of "direct", "threshold", "weighted"
        """
        return self.params["mode"]
    
    def get_params(self) -> Dict:
        """
        Return objective-specific parameters for display/logging.
        
        Returns:
            dict: Parameters dictionary (mode, threshold, or weight)
        
        Example:
            >>> computer = ObjectiveComputer({
            ...     "optuna_objective": "composite_min_f1_plur_corr",
            ...     "min_f1_threshold": 35.0
            ... })
            >>> params = computer.get_params()
            >>> assert params == {"mode": "threshold", "threshold": 35.0}
        """
        return self.params

