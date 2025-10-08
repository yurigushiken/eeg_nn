"""
Post-hoc analysis and refinement utilities.

This module provides post-hoc analysis tools that operate on trained models
and predictions without modifying training or model architecture.

Constitutional compliance:
- Section III (Deterministic Training): Post-hoc only, training unchanged
- Section IV (Rigorous Validation): Leak-free (tuned on inner, applied to outer)
- Section V (Audit-Ready Artifacts): All decisions traceable and reproducible

Modules:
    decision_layer: Ordinal adjacent-pair decision refinement
"""

