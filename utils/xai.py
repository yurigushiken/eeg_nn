"""
XAI utilities for EEG models.

Writes:
- Integrated Gradients (IG)                 → xai_analysis/integrated_gradients/
- IG topomaps (3 styles, per fold; robust) → xai_analysis/integrated_gradients_topomaps/
- Grad-CAM heatmaps                        → xai_analysis/gradcam_heatmaps/
- Grad-TopoCAM topomaps (3 styles)        → xai_analysis/gradcam_topomaps/

Note:
If Grad-CAM heatmaps show vertical bands only, choose an earlier conv with
`target_layer_name` so the layer still preserves [channels × time].
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple


def compute_and_plot_attributions(*args, **kwargs):
    return None


def compute_and_plot_gradcam(*args, **kwargs):
    return None


def compute_and_plot_gradcam_topomap(*args, **kwargs):
    return None
