"""
Figure 2: Nested Cross-Validation Structure (Publication Quality - V2)

Improvements based on PI feedback:
- Removed in-figure title
- Wong colorblind-safe palette for train/val/test
- Consistent box heights and spacing
- Grid alignment
- Simplified text
- 600 DPI export
"""

import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from pub_style import COLORS, save_publication_figure
from pathlib import Path

def create_improved_nested_cv():
    """Create publication-quality nested CV schematic"""
    
    fig = plt.figure(figsize=(10, 8), layout='constrained')
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 2.5, 2], hspace=0.35)
    
    # Consistent parameters
    box_height = 0.45
    subject_width = 0.85
    
    # ========== Panel A: All Subjects ==========
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 26)
    ax1.set_ylim(0, 2)
    ax1.axis('off')
    
    ax1.text(13, 1.7, 'All Subjects (N = 24)', ha='center', fontsize=10, weight='bold')
    
    # Draw subject boxes
    for i in range(24):
        rect = Rectangle((i + 0.6, 1), subject_width, box_height,
                        facecolor=COLORS['data'], edgecolor='#333', linewidth=0.8)
        ax1.add_patch(rect)
        ax1.text(i + 1.03, 1.225, f'S{i+1}', ha='center', va='center', 
                fontsize=6, color='white', weight='bold')
    
    # Arrow down with label
    arrow = FancyArrowPatch(
        (13, 0.85), (13, 0.4),
        arrowstyle='-|>', mutation_scale=12,
        linewidth=1.5, color='#555'
    )
    ax1.add_patch(arrow)
    ax1.text(13.6, 0.62, 'LOSO', fontsize=8, style='italic')
    
    # ========== Panel B: Outer Loop Examples ==========
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 26)
    ax2.set_ylim(0, 6.5)
    ax2.axis('off')
    
    ax2.text(13, 6.2, 'Outer Loop: LOSO (24 folds)', ha='center', fontsize=10, weight='bold')
    
    # Three example folds with consistent positioning
    fold_info = [
        (5.0, 'Fold 1', 0),    # Test S1
        (3.0, 'Fold 12', 11),  # Test S12
        (1.0, 'Fold 24', 23)   # Test S24
    ]
    
    for y_pos, fold_label, test_idx in fold_info:
        # Fold label
        ax2.text(0.3, y_pos + 0.225, fold_label, fontsize=9, weight='bold', va='center')
        
        # Draw subjects
        for i in range(24):
            x_pos = i + 1.5
            
            if i == test_idx:
                # Test subject (red)
                color = COLORS['test']
                label_txt = f'S{test_idx+1}'
                txt_color = 'black'
            else:
                # Train subjects (green)
                color = COLORS['train']
                idx_actual = i if i < test_idx else i + 1
                label_txt = f'{idx_actual+1}'
                txt_color = 'black'
            
            rect = Rectangle((x_pos, y_pos), subject_width, box_height,
                           facecolor=color, edgecolor='#333', linewidth=0.6)
            ax2.add_patch(rect)
            ax2.text(x_pos + subject_width/2, y_pos + box_height/2, label_txt,
                    ha='center', va='center', fontsize=5, color=txt_color)
        
        # Arrow for Fold 1 only (showing inner loop)
        if fold_label == 'Fold 1':
            arrow = FancyArrowPatch(
                (13, y_pos - 0.15), (13, y_pos - 0.65),
                arrowstyle='-|>', mutation_scale=10,
                linewidth=1.3, color='#555'
            )
            ax2.add_patch(arrow)
            ax2.text(13.5, y_pos - 0.4, 'Inner Loop\n(23 subjects)', 
                    fontsize=7, style='italic', va='center')
    
    # ========== Panel C: Inner Loop Detail ==========
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 26)
    ax3.set_ylim(0, 4)
    ax3.axis('off')
    
    ax3.text(13, 3.7, 'Inner Loop: 5-Fold GroupKFold (Model Selection)', 
            ha='center', fontsize=10, weight='bold')
    ax3.text(13, 3.35, '23 training subjects split for hyperparameter tuning',
            ha='center', fontsize=8, style='italic')
    
    # Draw 5 inner folds (simplified schematic)
    fold_width = 4.0
    fold_gap = 0.3
    start_x = 2.5
    
    for fold_i in range(5):
        y_pos = 2.4 - fold_i * 0.5
        
        # Fold label
        ax3.text(0.5, y_pos + 0.175, f'Inner\nFold {fold_i+1}', 
                fontsize=7, ha='center', va='center')
        
        # Draw 5 segments (4 train, 1 val)
        for seg_i in range(5):
            x_pos = start_x + seg_i * (fold_width + fold_gap)
            
            if seg_i == fold_i:
                # Validation segment (yellow)
                color = COLORS['validation']
                label_txt = 'VAL' if fold_i == 2 else ''  # Label middle one only
            else:
                # Training segments (green)
                color = COLORS['train']
                label_txt = 'TRAIN' if fold_i == 0 and seg_i == 0 else ''  # Label first one only
            
            rect = Rectangle((x_pos, y_pos), fold_width, 0.35,
                           facecolor=color, edgecolor='#333', linewidth=0.7)
            ax3.add_patch(rect)
            
            if label_txt:
                ax3.text(x_pos + fold_width/2, y_pos + 0.175, label_txt,
                        ha='center', va='center', fontsize=7, weight='bold')
    
    # Key insights box
    insights = [
        '• Subjects never overlap between train/validation/test',
        '• Augmentations applied ONLY to training data',
        '• Best model: argmax(inner_mean_min_per_class_f1)',
        '• Outer prediction: ensemble of K = 5 inner models'
    ]
    
    insight_text = '\n'.join(insights)
    ax3.text(0.74, 0.45, insight_text, transform=ax3.transAxes,
            fontsize=7, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8DC', 
                     edgecolor='#333', linewidth=1))
    
    # Legend
    legend_y = 0.3
    legend_elements = [
        (COLORS['train'], 'Training'),
        (COLORS['validation'], 'Validation (Inner)'),
        (COLORS['test'], 'Test (Outer)')
    ]
    
    legend_x_start = 2.5
    for i, (color, label) in enumerate(legend_elements):
        x_pos = legend_x_start + i * 6
        rect = Rectangle((x_pos, legend_y), 1.2, 0.25,
                        facecolor=color, edgecolor='#333', linewidth=0.8)
        ax3.add_patch(rect)
        ax3.text(x_pos + 1.5, legend_y + 0.125, label, 
                fontsize=8, va='center')
    
    return fig

if __name__ == '__main__':
    output_dir = Path('../../outputs/v2_improved')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING FIGURE 2: NESTED CV (V2 - PUBLICATION QUALITY)")
    print("="*60)
    
    fig = create_improved_nested_cv()
    
    # Save in publication formats
    save_publication_figure(fig, output_dir / 'figure2_nested_cv_v2',
                           formats=['pdf', 'png', 'svg'])
    
    plt.close()
    print("\n[OK] Figure 2 V2 complete - Publication ready!")
    print("="*60)

