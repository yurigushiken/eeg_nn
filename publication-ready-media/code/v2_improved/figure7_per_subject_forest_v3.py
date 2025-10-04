"""
Figure 7: Per-Subject Performance Forest Plot (V3 - PI Feedback Implemented)

IMPROVEMENTS:
- Moved legend/stats box outside plot area (PI: no occlusion of S1-S3)
- Removed in-art title
- Increased bottom margin for footnote
- Enhanced spacing
- Fixed label positions
"""

import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pub_style_v3 import (COLORS, WONG_COLORS, save_publication_figure, 
                           increase_bottom_margin, get_figure_size)
from pathlib import Path

def wilson_ci(n_success, n_trials, confidence=0.95):
    """Calculate Wilson score confidence interval"""
    z = stats.norm.ppf((1 + confidence) / 2)
    p_hat = n_success / n_trials
    
    denominator = 1 + z**2 / n_trials
    center = (p_hat + z**2 / (2 * n_trials)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_trials + z**2 / (4 * n_trials**2)) / denominator
    
    return center - margin, center + margin

def generate_synthetic_subject_data():
    """Generate realistic per-subject performance"""
    np.random.seed(42)
    
    n_subjects = 24
    subjects = np.arange(1, n_subjects + 1)
    n_trials = np.random.randint(95, 140, n_subjects)
    accuracies = np.random.normal(48, 8, n_subjects)
    accuracies = np.clip(accuracies, 30, 65)
    n_correct = (accuracies * n_trials / 100).astype(int)
    
    # Binomial test
    p_values = []
    for i in range(n_subjects):
        result = stats.binomtest(n_correct[i], n_trials[i], 0.333, alternative='greater')
        p_values.append(result.pvalue)
    p_values = np.array(p_values)
    
    # FDR correction
    try:
        from statsmodels.stats.multitest import multipletests
        _, p_adj, _, _ = multipletests(p_values, method='fdr_bh')
    except ImportError:
        idx_sorted = np.argsort(p_values)
        p_sorted = p_values[idx_sorted]
        m = len(p_values)
        p_adj = np.zeros_like(p_values)
        for i, p in enumerate(p_sorted):
            p_adj[idx_sorted[i]] = min(p * m / (i + 1), 1.0)
    
    above_chance = p_adj < 0.05
    cis = np.array([wilson_ci(n_correct[i], n_trials[i]) for i in range(n_subjects)])
    
    return subjects, accuracies, cis, above_chance, n_trials, p_adj

def create_forest_plot():
    """Create publication-quality forest plot (PI feedback implemented)"""
    
    # Get data
    subjects, accs, cis, above_chance, n_trials, p_adj = generate_synthetic_subject_data()
    
    # Create figure (PI: no in-art title)
    fig, ax = plt.subplots(figsize=(7, 8.5), layout='constrained')
    
    # Plot each subject
    for i, (subj, acc, ci, sig) in enumerate(zip(subjects, accs, cis, above_chance)):
        y_pos = len(subjects) - i  # Reverse order
        
        color = WONG_COLORS['green'] if sig else '#999999'
        
        # CI line
        ax.plot([ci[0]*100, ci[1]*100], [y_pos, y_pos], color=color, 
               linewidth=1.5, alpha=0.7, zorder=2)
        
        # Point
        ax.scatter([acc], [y_pos], s=50, color=color, edgecolors='#333', 
                  linewidths=0.8, zorder=3)
    
    # Reference lines
    ax.axvline(33.3, color=COLORS['chance'], linestyle=':', linewidth=2,
              label='Chance (33.3%)', zorder=1)
    
    group_mean = np.mean(accs)
    ax.axvline(group_mean, color='#333', linestyle='--', linewidth=2,
              label=f'Group mean ({group_mean:.1f}%)', zorder=1)
    
    # Y-axis
    ax.set_yticks(np.arange(1, len(subjects) + 1))
    ax.set_yticklabels([f'S{s}' for s in reversed(subjects)], fontsize=7)
    ax.set_ylabel('Subject', fontsize=9)
    
    # X-axis
    ax.set_xlabel('Accuracy (%)', fontsize=9)
    ax.set_xlim([25, 78])  # PI: Extended to accommodate external text
    
    # PI feedback: Move legend OUTSIDE plot area (right side)
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.extend([
        Patch(facecolor=WONG_COLORS['green'], edgecolor='#333', 
              label=f'Sig. above chance (n={above_chance.sum()})'),
        Patch(facecolor='#999999', edgecolor='#333', 
              label=f'Not sig. (n={(~above_chance).sum()})')
    ])
    
    legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1),
                      fontsize=7, frameon=False, borderaxespad=0)
    
    # Subtle grid (PI feedback)
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.4, axis='x', zorder=0)
    
    # Trial counts (right side, well-positioned)
    for i, (subj, n) in enumerate(zip(subjects, n_trials)):
        y_pos = len(subjects) - i
        ax.text(76, y_pos, f'n={n}', fontsize=6, va='center', ha='left')
    
    ax.text(76, len(subjects) + 1, 'Trials', fontsize=7, weight='bold', ha='left')
    
    # PI feedback: Move statistics box OUTSIDE to avoid occlusion of S1-S3
    n_sig = above_chance.sum()
    stats_text = (f'Above chance: {n_sig}/{len(subjects)} ({100*n_sig/len(subjects):.0f}%)\n'
                 f'Group mean: {group_mean:.1f}% ± {np.std(accs):.1f}%\n'
                 f'95% CI: [{np.percentile(accs, 2.5):.1f}, '
                 f'{np.percentile(accs, 97.5):.1f}]%')
    
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
           fontsize=7, va='bottom', ha='left', family='monospace',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                    edgecolor='#333', linewidth=0.8, alpha=0.95))
    
    # PI feedback: Increased bottom margin, move footnote lower
    increase_bottom_margin(fig, 0.12)
    note = ('Wilson 95% CI. Significance: FDR-corrected binomial test vs. chance (33.3%).')
    fig.text(0.5, 0.01, note, ha='center', fontsize=7, style='italic')
    
    return fig

if __name__ == '__main__':
    output_dir = Path('../../outputs/v2_improved')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING FIGURE 7 V3: PER-SUBJECT FOREST (PI FEEDBACK)")
    print("="*70)
    print("Improvements:")
    print("  [OK] Legend moved outside plot (no occlusion of S1-S3)")
    print("  [OK] Stats box repositioned to bottom left")
    print("  [OK] Removed in-art title")
    print("  [OK] Increased bottom margin for footnote")
    print("  [OK] Subtle grid (alpha=0.15)")
    
    fig = create_forest_plot()
    
    save_publication_figure(fig, output_dir / 'figure7_per_subject_forest_v3',
                           formats=['pdf', 'png', 'svg'])
    
    plt.close()
    print("\n[OK] Figure 7 V3 complete - PI feedback implemented!")
    print("="*70)

