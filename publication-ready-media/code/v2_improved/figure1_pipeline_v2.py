"""
Figure 1: End-to-End Pipeline (Publication Quality - V2)

Improvements based on PI feedback:
- Removed in-figure title (moved to caption)
- Flat colors from Wong colorblind-safe palette
- Consistent box sizes, spacing, alignment
- Simplified text (3 lines max per box)
- 600 DPI export with embedded fonts
- No gradients or shadows
"""

import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pub_style import COLORS, save_publication_figure, create_flowchart_box, create_arrow
from pathlib import Path

def create_improved_pipeline():
    """Create publication-quality pipeline flowchart"""
    
    # Figure setup - portrait for vertical flow
    fig, ax = plt.subplots(figsize=(7, 10), layout='constrained')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Box dimensions (consistent)
    box_width = 8
    box_height_std = 0.7
    box_height_tall = 1.0
    x_center = 1  # Left edge
    
    # Stage 1: Data Acquisition
    y_pos = 19
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_std,
        'Raw Data Acquisition\n24 subjects, ~360 trials/subject, 128-channel EEG',
        facecolor=COLORS['data'], fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.4))
    
    # Stage 2: Preprocessing
    y_pos -= 1.3
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_tall,
        'Preprocessing (HAPPE)\nBandpass 1.5–40 Hz · ICA artifact rejection\nBad channel interpolation · Epoch −200 to +800 ms',
        facecolor=COLORS['preprocessing'], fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.5))
    
    # Stage 3: Data Finalization
    y_pos -= 1.8
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_std,
        'Data Finalization & QC\nBehavioral alignment · Min trials/class · Channel intersection (100 ch)',
        facecolor=COLORS['data'], fontsize=8
    )
    create_arrow(ax, (5, y_pos), (5, y_pos - 0.4))
    
    # Stage 4: Task Split (side by side)
    y_pos -= 1.2
    create_flowchart_box(
        ax, (0.5, y_pos), 4, 0.6,
        'Cardinality 1–3\n(n = 3 classes)',
        facecolor=COLORS['evaluation'], fontsize=8
    )
    create_flowchart_box(
        ax, (5.5, y_pos), 4, 0.6,
        'Cardinality 4–6\n(n = 3 classes)',
        facecolor=COLORS['evaluation'], fontsize=8
    )
    # Arrows converging
    create_arrow(ax, (2.5, y_pos), (5, y_pos - 0.6))
    create_arrow(ax, (7.5, y_pos), (5, y_pos - 0.6))
    
    # Stage 5: Optuna Optimization (3-stage cluster)
    y_pos -= 1.8
    # Container box
    container = FancyBboxPatch(
        (0.3, y_pos - 1.5), 9.4, 2.2,
        boxstyle='round,pad=0.04',
        facecolor='#FFF5E6',  # Very light orange background
        edgecolor=COLORS['optimization'],
        linewidth=1.5,
        zorder=0
    )
    ax.add_patch(container)
    
    # Title for container
    ax.text(5, y_pos + 0.5, '3-Stage Hyperparameter Optimization (Optuna/TPE)',
           ha='center', fontsize=9, weight='bold')
    
    # Three stages
    stage_width = 2.6
    stage_y = y_pos - 0.6
    
    create_flowchart_box(
        ax, (0.7, stage_y), stage_width, 0.9,
        'Stage 1\nArchitecture\n~50 trials',
        facecolor=COLORS['optimization'], fontsize=8
    )
    ax.text(2, stage_y - 0.3, 'winner →', ha='center', fontsize=7, style='italic')
    
    create_flowchart_box(
        ax, (3.7, stage_y), stage_width, 0.9,
        'Stage 2\nLearning\n~50 trials',
        facecolor=COLORS['optimization'], fontsize=8
    )
    ax.text(5, stage_y - 0.3, 'winner →', ha='center', fontsize=7, style='italic')
    
    create_flowchart_box(
        ax, (6.7, stage_y), stage_width, 0.9,
        'Stage 3\nAugmentation\n~30 trials',
        facecolor=COLORS['optimization'], fontsize=8
    )
    
    # Objective note
    ax.text(5, y_pos - 1.3,
           'Objective: inner_mean_min_per_class_f1 (ensures all classes decodable)',
           ha='center', fontsize=7, style='italic')
    
    create_arrow(ax, (5, y_pos - 1.6), (5, y_pos - 2.1))
    
    # Stage 6: Final Evaluation
    y_pos -= 4.2
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_tall,
        'Final Evaluation (Multi-seed LOSO-CV)\n10 independent seeds · Leave-One-Subject-Out\nNested 5-fold inner CV · Ensemble predictions',
        facecolor=COLORS['evaluation'], fontsize=8
    )
    
    # Split to two paths
    create_arrow(ax, (3, y_pos), (3, y_pos - 0.5))
    create_arrow(ax, (7, y_pos), (7, y_pos - 0.5))
    
    # Stage 7: Parallel analysis (Stats & XAI)
    y_pos -= 1.7
    create_flowchart_box(
        ax, (0.5, y_pos), 4, 1.0,
        'Statistical Validation\nPermutation testing (n = 200)\nPer-subject significance\nMixed-effects modeling',
        facecolor=COLORS['statistics'], fontsize=8
    )
    create_flowchart_box(
        ax, (5.5, y_pos), 4, 1.0,
        'Explainable AI (XAI)\nIntegrated Gradients\nChannel importance\nSpatiotemporal patterns',
        facecolor=COLORS['statistics'], fontsize=8
    )
    
    # Converge to results
    create_arrow(ax, (2.5, y_pos), (5, y_pos - 0.6))
    create_arrow(ax, (7.5, y_pos), (5, y_pos - 0.6))
    
    # Stage 8: Results
    y_pos -= 1.6
    create_flowchart_box(
        ax, (x_center, y_pos), box_width, box_height_std,
        'Publication Results\nAccuracy 48–52% (chance 33.3%) · All classes decodable · p < 0.01',
        facecolor='#D4EDDA', fontsize=8  # Light green (success color)
    )
    
    return fig

if __name__ == '__main__':
    output_dir = Path('../../outputs/v2_improved')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING FIGURE 1: PIPELINE (V2 - PUBLICATION QUALITY)")
    print("="*60)
    
    fig = create_improved_pipeline()
    
    # Save in publication formats
    save_publication_figure(fig, output_dir / 'figure1_pipeline_v2', 
                           formats=['pdf', 'png', 'svg'])
    
    plt.close()
    print("\n[OK] Figure 1 V2 complete - Publication ready!")
    print("="*60)

