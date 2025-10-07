"""
Setup orchestrator for training runs.

This module handles all setup and initialization tasks for a training run:
- Logging setup (JSONL runtime log)
- Outer CV split computation (GroupKFold or LOSO)
- Channel topomap generation for scientific transparency
- Configuration validation

Constitutional compliance:
- Section III (Deterministic Training): Explicit seed handling
- Section IV (Subject-Aware CV): GroupKFold enforcement
- Section V (Audit-Ready Artifacts): Runtime logging

Classes:
    SetupOrchestrator: Coordinates setup tasks for training run
"""

from __future__ import annotations
from typing import Dict, List, Callable, Any
from pathlib import Path
import json
import numpy as np
from datetime import datetime, timezone
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold


class SetupOrchestrator:
    """
    Orchestrates setup and initialization for a training run.
    
    Responsibilities:
    - Setup runtime logging (JSONL format)
    - Compute outer CV splits (GroupKFold or LOSO)
    - Generate channel topomap for scientific transparency
    - Log configuration parameters
    
    Example usage:
        >>> orchestrator = SetupOrchestrator(cfg=cfg, run_dir=Path("results/run_123"))
        >>> log_fn = orchestrator.setup_logging()
        >>> outer_pairs = orchestrator.compute_outer_splits(
        ...     dataset=dataset,
        ...     y_all=y_all,
        ...     groups=groups,
        ...     predefined_splits=None,
        ... )
        >>> orchestrator.generate_channel_topomap()
    """
    
    def __init__(self, cfg: Dict, run_dir: Path | None):
        """
        Initialize SetupOrchestrator.
        
        Args:
            cfg: Configuration dictionary
            run_dir: Optional run directory for artifacts
        """
        self.cfg = cfg
        self.run_dir = run_dir
        self.jsonl_log_path = None
    
    def setup_logging(self) -> Callable:
        """
        Setup JSONL runtime logging.
        
        Returns:
            Logging function that can be called to log events
        """
        if not self.run_dir:
            # No run_dir, return no-op logger
            def _log_event_noop(event: str, message: str = "", **extra):
                pass
            return _log_event_noop
        
        # Create logs directory
        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_log_path = logs_dir / "runtime.jsonl"
        
        # Define logging function
        def _log_event(event: str, message: str = "", **extra):
            if self.jsonl_log_path:
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "level": "INFO",
                    "logger": "training_runner",
                    "event": event,
                    "run_id": self.run_dir.name if self.run_dir else "",
                    "message": message,
                    "extra": extra,
                }
                with self.jsonl_log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
        
        # Log run start
        _log_event("run_start", f"Starting run with seed={self.cfg.get('seed')}")
        
        return _log_event
    
    def log_configuration(self, log_fn: Callable, num_classes: int):
        """
        Log key configuration parameters.
        
        Args:
            log_fn: Logging function from setup_logging()
            num_classes: Number of classes in dataset
        """
        # Log chance level
        try:
            if num_classes > 0:
                chance = 100.0 / float(num_classes)
                log_fn("chance_level_computed", f"chance={chance:.2f}%", num_classes=num_classes)
        except Exception:
            pass
        
        # Log effective randomness controls
        try:
            outer_mode = "GroupKFold" if self.cfg.get("n_folds") else "LOSO"
            print(
                f"[config] seed={self.cfg.get('seed')} outer_mode={outer_mode}",
                flush=True,
            )
        except Exception:
            pass
    
    def compute_outer_splits(
        self,
        dataset,
        y_all: np.ndarray,
        groups: np.ndarray,
        predefined_splits: List[dict] | None,
    ) -> List[tuple]:
        """
        Compute outer CV splits (or use predefined splits).
        
        Args:
            dataset: Dataset object
            y_all: All labels
            groups: Subject group assignments
            predefined_splits: Optional predefined splits
        
        Returns:
            List of (train_idx, test_idx) tuples for each outer fold
        """
        outer_pairs: List[tuple] = []
        
        if predefined_splits:
            # Use predefined splits
            for rec in predefined_splits:
                outer_pairs.append((np.array(rec["outer_train_idx"]), np.array(rec["outer_test_idx"])))
        else:
            # Compute splits based on config
            if self.cfg.get("n_folds"):
                # GroupKFold
                k = int(self.cfg["n_folds"])
                gkf_outer = GroupKFold(n_splits=k)
                outer_pairs = [
                    (np.array(tr), np.array(te))
                    for tr, te in gkf_outer.split(np.zeros(len(dataset)), y_all, groups)
                ]
            else:
                # Leave-One-Subject-Out
                outer_pairs = [
                    (np.array(tr), np.array(te))
                    for tr, te in LeaveOneGroupOut().split(np.zeros(len(dataset)), y_all, groups)
                ]
        
        return outer_pairs
    
    def generate_channel_topomap(self):
        """
        Generate channel selection topomap for scientific transparency.
        
        This creates a visualization showing which EEG channels were used,
        which aids in scientific interpretation and reproducibility.
        """
        if not self.run_dir:
            return
        
        try:
            # Import visualization function
            from utils import channel_viz as _channel_viz
            save_channel_topomap_for_run = _channel_viz.save_channel_topomap_for_run
        except Exception:
            # Channel viz not available, skip silently
            return
        
        try:
            # Find montage file
            proj_root = Path(__file__).resolve().parents[2]
            montage_path = proj_root / "net" / "AdultAverageNet128_v1.sfp"
            
            if montage_path.exists():
                save_channel_topomap_for_run(self.cfg, self.run_dir, montage_path)
            else:
                print(f"[channel_viz] WARNING: Montage not found at {montage_path}", flush=True)
        except Exception as e:
            print(f"[channel_viz] WARNING: Could not generate channel topomap: {e}", flush=True)

