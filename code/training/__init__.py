"""
Training submodule for EEG neural network pipeline.

Contains extracted training logic from monolithic training_runner.py:
- metrics.py: ObjectiveComputer for metric computation
- checkpointing.py: CheckpointManager for checkpoint selection and early stopping
- inner_loop.py: InnerTrainer for inner fold training loop
- evaluation.py: OuterEvaluator for outer test evaluation

Constitutional compliance: Section III (Deterministic Training)
"""

__all__ = []

