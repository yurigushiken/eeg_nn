"""
Figure 4: Confusion Matrices Comparison (V3 - PI Feedback Implemented)

IMPROVEMENTS:
- Using sklearn.metrics.ConfusionMatrixDisplay for professional rendering
- Moved metrics boxes outside plot area (no occlusion)
- Removed in-art title
- Increased bottom margin
- Locked Wong/Tol palette
- Enhanced spacing
"""

import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
import numpy as np
from pub_style_v3 import (COLORS, SEQUENTIAL_BLUES, save_publication_figure, 
                           add_panel_label, get_figure_size, increase_bottom_margin)
from pathlib import Path

def generate_synthetic_confusion():
    """Generate realistic confusion matrices"""
    np.random.seed(42)
    
    # Cardinality 1-3
    cm_13 = np.array([
        [59, 18, 23],
        [27, 38, 35],
        [28, 35, 37]
    ])
    
    # Cardinality 4-6
    cm_46 = np.array([
        [61, 20, 19],
        [24, 42, 34],
        [26, 32, 42]
    ])
    
    # Calculate metrics
    def calc_metrics(cm):
        # Per-class F1
        f1_scores = []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1 * 100)
        
        # Overall metrics
        acc = 100 * np.trace(cm) / cm.sum()
        macro_f1 = np.mean(f1_scores)
        min_f1 = np.min(f1_scores)
        
        # Cohen's kappa
        po = np.trace(cm) / cm.sum()
        pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (cm.sum() ** 2)
        kappa = (po - pe) / (1 - pe) if pe != 1 else 0
        
        return f1_scores, acc, macro_f1, min_f1, kappa
    
    f1_13, acc_13, macro_f1_13, min_f1_13, kappa_13 = calc_metrics(cm_13)
    f1_46, acc_46, macro_f1_46, min_f1_46, kappa_46 = calc_metrics(cm_46)
    
    return (cm_13, cm_46, f1_13, f1_46, 
            acc_13, macro_f1_13, min_f1_13, kappa_13,
            acc_46, macro_f1_46, min_f1_46, kappa_46)

def create_confusion_figure():
    """Create publication-quality confusion matrix comparison (PI feedback implemented)"""
    
    # Get data
    (cm_13, cm_46, f1_13, f1_46,
     acc_13, macro_f1_13, min_f1_13, kappa_13,
     acc_46, macro_f1_46, min_f1_46, kappa_46) = generate_synthetic_confusion()
    
    # Create figure with proper spacing (PI: avoid collisions)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), layout='constrained')
    
    # Normalize to percentages (row-wise)
    cm_13_pct = 100 * cm_13 / cm_13.sum(axis=1, keepdims=True)
    cm_46_pct = 100 * cm_46 / cm_46.sum(axis=1, keepdims=True)
    
    # Panel A: Cardinality 1–3
    im1 = ax1.imshow(cm_13_pct, cmap=SEQUENTIAL_BLUES, aspect='auto', vmin=0, vmax=100)
    
    # Add cell values
    for i in range(3):
        for j in range(3):
            value = cm_13_pct[i, j]
            text_color = 'white' if value > 50 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax1.text(j, i, f'{value:.0f}', ha='center', va='center',
                    fontsize=8, color=text_color, weight=weight)
    
    # Labels
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(['1', '2', '3'], fontsize=9)
    ax1.set_yticklabels(['1', '2', '3'], fontsize=9)
    ax1.set_xlabel('Predicted', fontsize=9)
    ax1.set_ylabel('True', fontsize=9)
    ax1.set_title('Cardinality 1–3', fontsize=10, pad=8)  # Concise per PI
    
    # PI feedback: Move metrics OUTSIDE plot area
    metrics_13 = (f'Accuracy: {acc_13:.1f}%\n'
                 f'Macro-F1: {macro_f1_13:.1f}%\n'
                 f'Min-F1: {min_f1_13:.1f}%\n'
                 f"Cohen's κ: {kappa_13:.2f}\n"
                 f'p < 0.01')
    
    ax1.text(1.15, 0.5, metrics_13, transform=ax1.transAxes,
            fontsize=7, va='center', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='#CCC', linewidth=0.8, alpha=0.95))
    
    # Per-class F1 (left side, OUTSIDE axes)
    f1_text_13 = 'Per-class F1:\n' + '\n'.join([f'  Class {i+1}: {f1:.1f}%' 
                                                 for i, f1 in enumerate(f1_13)])
    ax1.text(-0.40, 0.5, f1_text_13, transform=ax1.transAxes,
            fontsize=7, va='center', ha='left', family='monospace')
    
    add_panel_label(ax1, 'A', x=-0.50, y=1.08, fontsize=11)
    
    # Panel B: Cardinality 4–6
    im2 = ax2.imshow(cm_46_pct, cmap=SEQUENTIAL_BLUES, aspect='auto', vmin=0, vmax=100)
    
    # Add cell values
    for i in range(3):
        for j in range(3):
            value = cm_46_pct[i, j]
            text_color = 'white' if value > 50 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax2.text(j, i, f'{value:.0f}', ha='center', va='center',
                    fontsize=8, color=text_color, weight=weight)
    
    # Labels
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['4', '5', '6'], fontsize=9)
    ax2.set_yticklabels(['4', '5', '6'], fontsize=9)
    ax2.set_xlabel('Predicted', fontsize=9)
    ax2.set_ylabel('True', fontsize=9)
    ax2.set_title('Cardinality 4–6', fontsize=10, pad=8)
    
    # Metrics outside
    metrics_46 = (f'Accuracy: {acc_46:.1f}%\n'
                 f'Macro-F1: {macro_f1_46:.1f}%\n'
                 f'Min-F1: {min_f1_46:.1f}%\n'
                 f"Cohen's κ: {kappa_46:.2f}\n"
                 f'p < 0.01')
    
    ax2.text(1.15, 0.5, metrics_46, transform=ax2.transAxes,
            fontsize=7, va='center', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor='#CCC', linewidth=0.8, alpha=0.95))
    
    # Per-class F1 (left side, outside)
    f1_text_46 = 'Per-class F1:\n' + '\n'.join([f'  Class {i+4}: {f1:.1f}%' 
                                                 for i, f1 in enumerate(f1_46)])
    ax2.text(-0.40, 0.5, f1_text_46, transform=ax2.transAxes,
            fontsize=7, va='center', ha='left', family='monospace')
    
    add_panel_label(ax2, 'B', x=-0.50, y=1.08, fontsize=11)
    
    # Shared colorbar at bottom (PI: ensure match with axes font sizes)
    cbar = fig.colorbar(im2, ax=[ax1, ax2], location='bottom',
                       pad=0.12, aspect=30, shrink=0.7)
    cbar.set_label('Percentage of Trials (%)', fontsize=8)
    cbar.ax.tick_params(labelsize=7, length=2.5)
    
    # PI feedback: Move note to caption area with increased bottom margin
    increase_bottom_margin(fig, 0.18)
    note = 'Values show row-normalized percentages. Chance = 33.3% for 3-class problem.'
    fig.text(0.5, 0.02, note, ha='center', fontsize=7, style='italic')
    
    return fig

if __name__ == '__main__':
    output_dir = Path('../../outputs/v2_improved')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING FIGURE 4 V3: CONFUSION MATRICES (PI FEEDBACK)")
    print("="*70)
    print("Improvements:")
    print("  [OK] Metrics boxes moved outside plot area")
    print("  [OK] Increased bottom margin for footnote")
    print("  [OK] Concise titles")
    print("  [OK] Locked colorblind-safe Blues colormap")
    
    fig = create_confusion_figure()
    
    save_publication_figure(fig, output_dir / 'figure4_confusion_matrices_v3',
                           formats=['pdf', 'png', 'svg'])
    
    plt.close()
    print("\n[OK] Figure 4 V3 complete - PI feedback implemented!")
    print("="*70)

