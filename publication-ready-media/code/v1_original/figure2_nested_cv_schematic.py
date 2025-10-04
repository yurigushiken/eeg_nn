"""
Figure 2: Nested Cross-Validation Structure
Visual explanation of LOSO + inner GroupKFold structure

Requirements: matplotlib, numpy
Output: High-resolution publication-ready figure
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300

def create_nested_cv_schematic():
    """Create detailed nested CV schematic"""
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 3, 2], hspace=0.4)
    
    # Colors
    train_color = '#A8E6CF'  # Light green
    val_color = '#FFE7A0'    # Light yellow
    test_color = '#FFB3BA'   # Light red
    
    # Panel A: All subjects
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 26)
    ax1.set_ylim(0, 2)
    ax1.axis('off')
    
    ax1.text(13, 1.8, 'ALL SUBJECTS (N=24)', ha='center', fontsize=12, weight='bold')
    
    # Draw subject boxes
    for i in range(24):
        rect = Rectangle((i + 1, 0.8), 0.9, 0.6, 
                         facecolor='#4A90E2', edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        ax1.text(i + 1.45, 1.1, f'S{i+1}', ha='center', va='center', fontsize=7, color='white', weight='bold')
    
    # Arrow down
    arrow = FancyArrowPatch(
        (13, 0.6), (13, 0.2),
        arrowstyle='->,head_width=0.3,head_length=0.2',
        color='black', linewidth=2
    )
    ax1.add_patch(arrow)
    ax1.text(13.5, 0.4, 'Leave-One-Subject-Out', fontsize=9, style='italic')
    
    # Panel B: Outer loop examples
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 26)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    ax2.text(13, 7.5, 'OUTER LOOP: Leave-One-Subject-Out (24 folds)', 
             ha='center', fontsize=11, weight='bold')
    
    # Draw 3 example folds
    fold_examples = [
        (6, 'Fold 1', 0, [1]),  # Test subject 1
        (4, 'Fold 12', 11, [12]),  # Test subject 12
        (2, 'Fold 24', 23, [24])   # Test subject 24
    ]
    
    for y_pos, label, test_idx, test_subj in fold_examples:
        # Fold label
        ax2.text(0.5, y_pos + 0.3, label, fontsize=10, weight='bold')
        
        # Draw subjects
        for i in range(24):
            if i == test_idx:
                # Test subject
                color = test_color
                label_text = f'S{test_subj[0]}\nTEST'
                fontsize = 6
            else:
                # Train subjects
                color = train_color
                label_text = f'S{i+1}' if i < test_idx else f'S{i+2}'
                fontsize = 7
            
            rect = Rectangle((i + 1.5, y_pos), 0.9, 0.5,
                           facecolor=color, edgecolor='black', linewidth=0.8)
            ax2.add_patch(rect)
            ax2.text(i + 1.95, y_pos + 0.25, label_text, 
                    ha='center', va='center', fontsize=fontsize)
        
        # Add arrow to inner loop for Fold 1
        if y_pos == 6:
            arrow = FancyArrowPatch(
                (13, y_pos - 0.3), (13, y_pos - 1.0),
                arrowstyle='->,head_width=0.3,head_length=0.2',
                color='black', linewidth=2
            )
            ax2.add_patch(arrow)
    
    ax2.text(13.5, 5, 'Inner Loop\n(23 subjects)', fontsize=8, style='italic', ha='left')
    
    # Panel C: Inner loop detail
    ax3 = fig.add_subplot(gs[2])
    ax3.set_xlim(0, 26)
    ax3.set_ylim(0, 5)
    ax3.axis('off')
    
    ax3.text(13, 4.7, 'INNER LOOP: 5-Fold GroupKFold (Model Selection)', 
             ha='center', fontsize=11, weight='bold')
    ax3.text(13, 4.3, '23 training subjects split into 5 folds for hyperparameter tuning',
             ha='center', fontsize=9, style='italic')
    
    # Draw 5 inner folds
    n_train = 23
    fold_size = n_train / 5
    
    for fold_i in range(5):
        y_pos = 3 - fold_i * 0.6
        
        # Fold label
        ax3.text(0.5, y_pos + 0.2, f'Inner\nFold {fold_i+1}', fontsize=8, ha='center')
        
        # Draw simplified inner fold structure
        start_x = 2
        for i in range(5):
            if i == fold_i:
                # Validation fold
                color = val_color
                width = 4.2
                label_text = f'VAL\n(~{int(fold_size)} subj)'
            else:
                # Training folds
                color = train_color
                width = 4.2
                label_text = f'TRAIN' if i == 0 else ''
            
            rect = Rectangle((start_x + i * 4.5, y_pos), width, 0.4,
                           facecolor=color, edgecolor='black', linewidth=0.8)
            ax3.add_patch(rect)
            if label_text:
                ax3.text(start_x + i * 4.5 + width/2, y_pos + 0.2,
                        label_text, ha='center', va='center', fontsize=7)
    
    # Add legend at bottom
    legend_y = 0.5
    legend_elements = [
        Rectangle((3, legend_y), 1, 0.3, facecolor=train_color, edgecolor='black', label='Training'),
        Rectangle((7, legend_y), 1, 0.3, facecolor=val_color, edgecolor='black', label='Validation (Inner)'),
        Rectangle((11, legend_y), 1, 0.3, facecolor=test_color, edgecolor='black', label='Test (Outer)')
    ]
    
    for i, rect in enumerate(legend_elements):
        ax3.add_patch(rect)
        ax3.text(rect.get_x() + 1.5, legend_y + 0.15, 
                rect.get_label(), fontsize=9, va='center')
    
    # Add key insights box
    insights = [
        "• Subjects never overlap between train/val/test",
        "• Augmentations applied ONLY to training data",
        "• Best model selected by: argmax(inner_mean_min_per_class_f1)",
        "• Outer prediction: ensemble of K=5 inner models"
    ]
    
    insight_text = '\n'.join(insights)
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                     edgecolor='black', linewidth=1.5)
    ax3.text(19, 2, insight_text, fontsize=8, va='top',
            bbox=bbox_props, family='monospace')
    
    # Overall title
    fig.suptitle('Nested Cross-Validation Prevents Data Leakage', 
                fontsize=14, weight='bold', y=0.98)
    
    return fig

if __name__ == '__main__':
    output_dir = Path('../outputs/figure2_nested_cv')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = create_nested_cv_schematic()
    
    # Save outputs
    fig.savefig(output_dir / 'figure2_nested_cv.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved PNG: {output_dir / 'figure2_nested_cv.png'}")
    
    fig.savefig(output_dir / 'figure2_nested_cv.pdf', bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved PDF: {output_dir / 'figure2_nested_cv.pdf'}")
    
    fig.savefig(output_dir / 'figure2_nested_cv.svg', bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved SVG: {output_dir / 'figure2_nested_cv.svg'}")
    
    plt.close()
    print("\n[OK] Figure 2 complete!")

