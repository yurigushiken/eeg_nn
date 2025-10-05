"""
Figure 2: Nested CV (V4 - Neuroscience Standards)
- Fixed text box occlusion (bottom right)
- Conservative colors
- Clear spacing
"""

import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pub_style_v4 import COLORS, save_publication_figure
from pathlib import Path

def create_nested_cv_v4():
    """Create nested CV schematic with no occlusions and real subject IDs."""
    
    # Real subject IDs (24 subjects, but IDs go up to 33 with gaps)
    real_subject_ids = ['02', '03', '04', '05', '08', '09', '10', '11', '12', '13', 
                        '14', '15', '17', '21', '22', '23', '25', '26', '27', '28', 
                        '29', '31', '32', '33']
    
    fig = plt.figure(figsize=(10, 9.2))  # Increased height to prevent occlusions
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 2.5, 2.5], hspace=0.35)
    
    box_height = 0.45
    subject_width = 0.85
    
    # ========== Panel A: All Subjects ==========
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 26)
    ax1.set_ylim(0, 2)
    ax1.axis('off')
    
    ax1.text(13, 1.7, 'All Subjects (N = 24)', ha='center', fontsize=10, weight='bold')
    
    for i in range(24):
        rect = Rectangle((i + 0.6, 1), subject_width, box_height,
                        facecolor=COLORS['data'], edgecolor='#333', linewidth=0.8)
        ax1.add_patch(rect)
        ax1.text(i + 1.03, 1.225, f'S{real_subject_ids[i]}', ha='center', va='center',
                fontsize=6, color='white', weight='bold')
    
    arrow = FancyArrowPatch(
        (13, 0.85), (13, 0.4),
        arrowstyle='-|>', mutation_scale=12,
        linewidth=1.5, color='#555'
    )
    ax1.add_patch(arrow)
    ax1.text(13.6, 0.62, 'LOSO', fontsize=8, style='italic')
    
    # ========== Panel B: Outer Loop ==========
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 26)
    ax2.set_ylim(0, 7.0)  # Increased for more space
    ax2.axis('off')
    
    ax2.text(13, 6.7, 'Outer Loop: LOSO (24 folds)', ha='center', fontsize=10, weight='bold')
    
    fold_info = [
        (5.5, 'Fold 1', 0),     # Moved up
        (3.5, 'Fold 12', 11),   # Moved up
        (1.5, 'Fold 24', 23)    # Moved up
    ]
    
    for y_pos, fold_label, test_idx in fold_info:
        ax2.text(0.5, y_pos + 0.225, fold_label, fontsize=9, weight='bold', va='center')  # Moved right slightly
        
        for i in range(24):
            x_pos = i + 1.8  # Shifted right to give more space for labels
            
            if i == test_idx:
                color = COLORS['test']
                label = 'Test'
            else:
                color = COLORS['train']
                label = 'Train'
            
            rect = Rectangle((x_pos, y_pos), subject_width, box_height,
                            facecolor=color, edgecolor='#333', linewidth=0.7)
            ax2.add_patch(rect)
            ax2.text(x_pos + subject_width/2, y_pos + box_height/2,
                    f'S{real_subject_ids[i]}', ha='center', va='center',
                    fontsize=5.5, color='#333', weight='bold')
    
    # Legend (top right, clear)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['train'], ec='#333', lw=0.7),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['test'], ec='#333', lw=0.7)
    ]
    ax2.legend(legend_elements, ['Train (outer)', 'Test (outer)'],
              loc='upper right', fontsize=8, ncol=2, frameon=False)
    
    # Arrow to inner loop
    arrow_mid = FancyArrowPatch(
        (13, 1.0), (13, 0.1),
        arrowstyle='-|>', mutation_scale=12,
        linewidth=1.5, color='#555'
    )
    ax2.add_patch(arrow_mid)
    
    # ========== Panel C: Inner Loop ==========
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 26)
    ax3.set_ylim(0, 6.2)  # Increased for more bottom space to prevent text occlusion
    ax3.axis('off')
    
    ax3.text(13, 5.9, 'Inner Loop: 5-fold CV on Outer-Train (Fold 1 example)',
            ha='center', fontsize=10, weight='bold')
    
    n_inner_subjects = 23
    inner_folds = [
        (4.8, 'Inner Fold 1', [0, 1, 2, 3]),    # Moved up significantly
        (3.8, 'Inner Fold 2', [4, 5, 6, 7]),    # Moved up
        (2.8, 'Inner Fold 3', [8, 9, 10, 11]),  # Moved up
        (1.8, 'Inner Fold 4', [12, 13, 14, 15]), # Moved up
        (0.8, 'Inner Fold 5', [16, 17, 18, 19])  # Moved up to prevent occlusion
    ]
    
    for y_pos, fold_label, val_indices in inner_folds:
        ax3.text(0.2, y_pos + 0.15, fold_label, fontsize=7.5, va='center')
        
        for i in range(n_inner_subjects):
            x_pos = i + 1.5
            
            if i in val_indices:
                color = COLORS['validation']
            else:
                color = COLORS['train']
            
            rect = Rectangle((x_pos, y_pos), subject_width * 0.95, 0.35,
                            facecolor=color, edgecolor='#333', linewidth=0.6)
            ax3.add_patch(rect)
    
    # Legend (top right, clear positioning)
    legend_elements_inner = [
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['train'], ec='#333', lw=0.6),
        plt.Rectangle((0, 0), 1, 1, fc=COLORS['validation'], ec='#333', lw=0.6)
    ]
    ax3.legend(legend_elements_inner, ['Train (inner)', 'Validation (inner)'],
              loc='upper right', fontsize=8, ncol=2, frameon=False)
    
    # Note at BOTTOM with adequate clearance
    ax3.text(13, 0.0, 'Objective: composite (65% min F1 + 35% diagonal dominance) — averaged across inner folds',
            ha='center', fontsize=7, style='italic', color='#555')
    
    return fig

if __name__ == '__main__':
    output_dir = Path('D:/eeg_nn/publication-ready-media/outputs/v4')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING FIGURE 2 V4: NESTED CV (FIXED OCCLUSION)")
    print("="*70)
    print("V4 Improvements:")
    print("  [OK] Fixed text box occlusion (bottom right)")
    print("  [OK] Increased bottom spacing")
    print("  [OK] Clear layout, no overlaps")
    
    fig = create_nested_cv_v4()
    
    save_publication_figure(fig, output_dir / 'figure2_nested_cv_v4',
                           formats=['pdf', 'png', 'svg'])
    
    plt.close()
    print("\n[OK] Figure 2 V4 complete!")
    print("="*70)


