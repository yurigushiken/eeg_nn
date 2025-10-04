"""
Figure 1: Pipeline (V4 - Neuroscience Standards)
- MUTED BLUES ONLY (conservative palette)
- No occlusions
- White background
- High contrast text
"""

import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pub_style_v4 import COLORS, BLUES_PALETTE, save_publication_figure
from pathlib import Path

def create_flowchart_box(ax, xy, width, height, text, facecolor='lightblue',
                         edgecolor='#333', linewidth=1.0, fontsize=8):
    """Creates a flowchart box."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle='round,pad=0.05',
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth
    )
    ax.add_patch(box)
    
    text_x = xy[0] + width / 2
    text_y = xy[1] + height / 2
    ax.text(text_x, text_y, text, ha='center', va='center',
            fontsize=fontsize, color='#222', weight='normal')

def create_arrow(ax, start, end, color='#555', linewidth=1.5):
    """Creates a flowchart arrow."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=12,
        linewidth=linewidth,
        color=color
    )
    ax.add_patch(arrow)

def create_pipeline_v4():
    """Create muted blue pipeline flowchart."""
    
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    box_width = 8
    box_height_std = 0.7
    box_height_tall = 1.0
    x_center = 1
    
    # Stage 1: Data Acquisition (light blue)
    y_pos = 19
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_std,
        'Raw Data Acquisition\n24 subjects, ~360 trials/subject, 128-channel EEG',
        facecolor=BLUES_PALETTE['light'], fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.4))
    
    # Stage 2: Preprocessing (medium blue)
    y_pos -= 1.3
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_tall,
        'Preprocessing (HAPPE)\nBandpass 1.5–40 Hz · ICA artifact rejection\nBad channel interpolation · Epoch −200 to +800 ms',
        facecolor=BLUES_PALETTE['medium'], fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.5))
    
    # Stage 3: Data Finalization (medium-light blue)
    y_pos -= 1.8
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_std,
        'Data Finalization & QC\nBehavioral alignment · Min trials/class · Channel intersection (100 ch)',
        facecolor=BLUES_PALETTE['medium_light'], fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.4))
    
    # Stage 4: Task Split (very light blue)
    y_pos -= 1.2
    create_flowchart_box(
        ax, (0.5, y_pos), 4, 0.6,
        'Cardinality 1–3\n(n = 3 classes)',
        facecolor=BLUES_PALETTE['very_light'], fontsize=8
    )
    create_flowchart_box(
        ax, (5.5, y_pos), 4, 0.6,
        'Cardinality 4–6\n(n = 3 classes)',
        facecolor=BLUES_PALETTE['very_light'], fontsize=8
    )
    create_arrow(ax, (2.5, y_pos), (5, y_pos - 0.6))
    create_arrow(ax, (7.5, y_pos), (5, y_pos - 0.6))
    
    # Stage 5: Optuna (3-stage - medium-dark blues)
    y_pos -= 1.8
    # Outer box
    outer_box = FancyBboxPatch(
        (0.8, y_pos - 0.6), 8.4, 1.5,
        boxstyle='round,pad=0.08',
        facecolor='#F5F5F5',
        edgecolor='#666',
        linewidth=1.2
    )
    ax.add_patch(outer_box)
    
    ax.text(5, y_pos + 0.75, '3-Stage Hyperparameter Optimization (Optuna/TPE)',
            ha='center', fontsize=8, weight='bold')
    
    # Three stages
    stage_width = 2.6
    stages = [
        (1.2, 'Stage 1\nArchitecture\n~50 trials'),
        (4.1, 'Stage 2\nLearning\n~50 trials'),
        (7.0, 'Stage 3\nAugmentation\n~30 trials')
    ]
    for x_pos, label in stages:
        create_flowchart_box(
            ax, (x_pos, y_pos - 0.45), stage_width, 0.85,
            label, facecolor=BLUES_PALETTE['medium_dark'], fontsize=7
        )
    
    ax.text(5, y_pos - 1.15, 'Objective: inner_mean_min_per_class_f1 (ensures all classes decodable)',
            ha='center', fontsize=7, style='italic', color='#555')
    
    create_arrow(ax, (5, y_pos - 0.7), (5, y_pos - 1.4))
    
    # Stage 6: Final Evaluation (dark blue)
    y_pos -= 3.2
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_tall,
        'Final Evaluation (Multi-seed LOSO-CV)\n10 independent seeds · Leave-One-Subject-Out\nNested 5-fold inner CV · Ensemble predictions',
        facecolor=BLUES_PALETTE['dark'], fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.5))
    
    # Stage 7: Analysis (split - very dark blue)
    y_pos -= 1.8
    create_flowchart_box(
        ax, (0.5, y_pos), 4, 0.9,
        'Statistical Validation\nPermutation testing (n = 200)\nPer-subject significance\nMixed-effects modeling',
        facecolor=BLUES_PALETTE['very_dark'], fontsize=7
    )
    create_flowchart_box(
        ax, (5.5, y_pos), 4, 0.9,
        'Explainable AI (XAI)\nIntegrated Gradients\nChannel importance\nSpatiotemporal patterns',
        facecolor=BLUES_PALETTE['very_dark'], fontsize=7
    )
    create_arrow(ax, (2.5, y_pos), (5, y_pos - 0.6))
    create_arrow(ax, (7.5, y_pos), (5, y_pos - 0.6))
    
    # Stage 8: Results (deepest blue)
    y_pos -= 1.8
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, 0.7,
        'Publication Results\nAccuracy 48–52% (chance 33.3%) · All classes decodable · p < 0.01',
        facecolor=BLUES_PALETTE['deepest'], fontsize=8
    )
    
    return fig

if __name__ == '__main__':
    output_dir = Path('D:/eeg_nn/publication-ready-media/outputs/v4')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING FIGURE 1 V4: PIPELINE (MUTED BLUES)")
    print("="*70)
    print("V4 Improvements:")
    print("  [OK] Muted blues only (conservative palette)")
    print("  [OK] No bright colors")
    print("  [OK] Neuroscience publication standards")
    
    fig = create_pipeline_v4()
    
    save_publication_figure(fig, output_dir / 'figure1_pipeline_v4',
                           formats=['pdf', 'png', 'svg'])
    
    plt.close()
    print("\n[OK] Figure 1 V4 complete!")
    print("="*70)


