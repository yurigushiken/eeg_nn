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
    
    # Stage 1: Data Acquisition (lighter blue)
    y_pos = 19
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_std,
        'Raw Data Acquisition\n24 subjects, ~300 trials/subject, 128-channel EEG',
        facecolor='#D6E9F5', fontsize=8  # Lighter uniform blue
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.4))
    
    # Stage 2: Preprocessing (lighter blue)
    y_pos -= 1.3
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_tall,
        'Preprocessing (HAPPE)\n18 datasets: 3 HPF (1.0/1.5/2.0 Hz) × 3 LPF (35/40/45 Hz) × 2 baseline (on/off)\nICA artifact rejection · Bad channel interpolation · Epoch −200 to +800 ms',
        facecolor='#B8D9EE', fontsize=7.5
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.5))
    
    # Stage 3: Data Finalization (lighter blue)
    y_pos -= 1.8
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_std,
        'Data Finalization & QC\nBehavioral & condition alignment · Min trials/class · Channel intersection (~100 ch)',
        facecolor='#A8D0EA', fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.4))
    
    # Stage 4: Task Split (lighter blue)
    y_pos -= 1.2
    create_flowchart_box(
        ax, (0.5, y_pos), 4, 0.6,
        'Cardinality 1–3\n(n = 3 classes)',
        facecolor='#E8F3FA', fontsize=8
    )
    create_flowchart_box(
        ax, (5.5, y_pos), 4, 0.6,
        'Cardinality 4–6\n(n = 3 classes)',
        facecolor='#E8F3FA', fontsize=8
    )
    create_arrow(ax, (2.5, y_pos), (5, y_pos - 0.6))
    create_arrow(ax, (7.5, y_pos), (5, y_pos - 0.6))
    
    # Stage 5: Optuna (4-stage - lighter blues)
    y_pos -= 1.8
    # Outer box - taller for 4 stages
    outer_box = FancyBboxPatch(
        (0.6, y_pos - 0.6), 8.8, 1.7,
        boxstyle='round,pad=0.08',
        facecolor='#F5F5F5',
        edgecolor='#666',
        linewidth=1.2
    )
    ax.add_patch(outer_box)
    
    ax.text(5, y_pos + 0.95, '4-Stage Hyperparameter Optimization (Optuna/TPE)',
            ha='center', fontsize=8, weight='bold')
    
    # Four stages - narrower boxes to fit
    stage_width = 2.0
    stages = [
        (0.9, 'Stage 1\nArchitecture\n~50 trials'),
        (3.1, 'Stage 2\nSanity Check\n~50 trials'),
        (5.3, 'Stage 3\nRecipe\n~50 trials'),
        (7.5, 'Stage 4\nAugmentation\n~30 trials')
    ]
    for x_pos, label in stages:
        create_flowchart_box(
            ax, (x_pos, y_pos - 0.45), stage_width, 0.85,
            label, facecolor='#8FC5E8', fontsize=6.5  # Lighter uniform blue
        )
    
    # Objective text - moved down more for clearance from boxes
    ax.text(5, y_pos - 1.40, 'Objective: composite (65% min F1 + 35% plurality correctness) — ensures decodability & distinctness',
            ha='center', fontsize=6.5, style='italic', color='#555')
    
    # Arrow - positioned lower to avoid objective text
    create_arrow(ax, (5, y_pos - 0.7), (5, y_pos - 1.70))
    
    # Stage 6: Final Evaluation (lighter blue)
    y_pos -= 3.2
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_tall,
        'Final Evaluation (Multi-seed LOSO-CV)\n10 independent seeds · Leave-One-Subject-Out\nNested 5-fold inner CV · Refit predictions',
        facecolor='#7AB8DD', fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.5))
    
    # Stage 7: Statistical Validation (lighter blue, on its own row)
    y_pos -= 1.4
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, 0.9,
        'Statistical Validation\nPermutation testing (n = 200) · Per-subject significance · Mixed-effects modeling',
        facecolor='#6AAFD1', fontsize=7.5
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.4))
    
    # Stage 8: Explainable AI (lighter blue, on its own row)
    y_pos -= 1.2
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, 0.9,
        'Explainable AI (XAI)\nIntegrated Gradients · Channel importance · Spatiotemporal patterns',
        facecolor='#5AA6C5', fontsize=7.5
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.4))
    
    # Stage 9: Results (lighter blue)
    y_pos -= 1.2
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, 0.7,
        'Publication Results\nAccuracy 48–52% (chance 33.3%) · All classes decodable · p < 0.01',
        facecolor='#4A9DBF', fontsize=8
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


