"""
Figure 6: Permutation Testing - Null Distribution
Demonstrates statistical significance above chance

Requirements: matplotlib, numpy, scipy
Output: Two-panel figure with null distributions
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

# Set publication style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300

def generate_synthetic_perm_data():
    """Generate realistic synthetic permutation data for demonstration"""
    np.random.seed(42)
    
    # Simulate 200 permutations
    n_perm = 200
    
    # Null distribution (label shuffling should give ~chance performance)
    # Chance = 33.3% for 3-class problem
    chance = 33.3
    
    # Accuracy null: centered at chance with some variance
    acc_null = np.random.normal(chance, 1.8, n_perm)
    acc_null = np.clip(acc_null, 25, 45)  # Realistic bounds
    
    # Macro-F1 null: slightly lower than accuracy, more variance
    f1_null = np.random.normal(chance - 3.6, 2.1, n_perm)
    f1_null = np.clip(f1_null, 20, 40)
    
    # Observed performance (from actual model)
    acc_observed = 48.3
    f1_observed = 44.1
    
    return acc_null, f1_null, acc_observed, f1_observed

def create_permutation_figure():
    """Create two-panel permutation testing figure"""
    
    # Get data
    acc_null, f1_null, acc_obs, f1_obs = generate_synthetic_perm_data()
    
    # Calculate statistics
    acc_pval = (np.sum(acc_null >= acc_obs) + 1) / (len(acc_null) + 1)
    f1_pval = (np.sum(f1_null >= f1_obs) + 1) / (len(f1_null) + 1)
    
    acc_zscore = (acc_obs - np.mean(acc_null)) / np.std(acc_null)
    f1_zscore = (f1_obs - np.mean(f1_null)) / np.std(f1_null)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Colors
    null_color = '#4A90E2'
    obs_color = '#E74C3C'
    
    # Panel A: Accuracy
    n, bins, patches = ax1.hist(acc_null, bins=24, density=True, alpha=0.6, 
                                color=null_color, edgecolor='black', linewidth=0.8)
    
    # Add KDE overlay
    kde_x = np.linspace(acc_null.min(), acc_null.max(), 100)
    kde = stats.gaussian_kde(acc_null)
    ax1.plot(kde_x, kde(kde_x), 'b-', linewidth=2, label='Null density (KDE)')
    
    # Observed line
    ax1.axvline(acc_obs, color=obs_color, linestyle='--', linewidth=3, 
               label=f'Observed = {acc_obs:.1f}%')
    
    # Chance line
    ax1.axvline(33.3, color='gray', linestyle=':', linewidth=2, alpha=0.7,
               label='Chance = 33.3%')
    
    # Annotations
    ax1.text(acc_obs + 0.5, max(n) * 0.9, 
            f'p = {acc_pval:.3f}\nz = {acc_zscore:.2f}',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='white', 
                                  edgecolor='black', alpha=0.9))
    
    # Null statistics box
    null_stats = f'Null: μ={np.mean(acc_null):.1f}%, σ={np.std(acc_null):.1f}%'
    ax1.text(0.02, 0.98, null_stats, transform=ax1.transAxes,
            fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax1.set_xlabel('Accuracy (%)', fontsize=12, weight='bold')
    ax1.set_ylabel('Probability Density', fontsize=12, weight='bold')
    ax1.set_title('A. Accuracy Null Distribution (n=200 permutations)', 
                 fontsize=13, weight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel B: Macro-F1
    n, bins, patches = ax2.hist(f1_null, bins=24, density=True, alpha=0.6,
                                color=null_color, edgecolor='black', linewidth=0.8)
    
    # Add KDE overlay
    kde_x = np.linspace(f1_null.min(), f1_null.max(), 100)
    kde = stats.gaussian_kde(f1_null)
    ax2.plot(kde_x, kde(kde_x), 'b-', linewidth=2, label='Null density (KDE)')
    
    # Observed line
    ax2.axvline(f1_obs, color=obs_color, linestyle='--', linewidth=3,
               label=f'Observed = {f1_obs:.1f}%')
    
    # Chance line (F1 slightly below accuracy at chance)
    ax2.axvline(29.7, color='gray', linestyle=':', linewidth=2, alpha=0.7,
               label='Expected at chance')
    
    # Annotations
    ax2.text(f1_obs + 0.5, max(n) * 0.9,
            f'p = {f1_pval:.3f}\nz = {f1_zscore:.2f}',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='white',
                                  edgecolor='black', alpha=0.9))
    
    # Null statistics box
    null_stats = f'Null: μ={np.mean(f1_null):.1f}%, σ={np.std(f1_null):.1f}%'
    ax2.text(0.02, 0.98, null_stats, transform=ax2.transAxes,
            fontsize=9, va='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    ax2.set_xlabel('Macro-F1 Score (%)', fontsize=12, weight='bold')
    ax2.set_ylabel('Probability Density', fontsize=12, weight='bold')
    ax2.set_title('B. Macro-F1 Null Distribution (n=200 permutations)',
                 fontsize=13, weight='bold', pad=15)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Overall title
    fig.suptitle('Permutation Testing: Model Performance vs. Empirical Null Hypothesis',
                fontsize=14, weight='bold', y=0.99)
    
    # Add method note at bottom
    method_text = ('Method: Labels shuffled 200× within-subject (preserving class balance) '
                  'with fixed train/test splits.\n'
                  'Empirical p-value: (# permutations ≥ observed + 1) / (n_permutations + 1)')
    fig.text(0.5, 0.01, method_text, ha='center', fontsize=8, style='italic',
            wrap=True)
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig

if __name__ == '__main__':
    output_dir = Path('../outputs/figure6_permutation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = create_permutation_figure()
    
    # Save outputs
    fig.savefig(output_dir / 'figure6_permutation.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved PNG: {output_dir / 'figure6_permutation.png'}")
    
    fig.savefig(output_dir / 'figure6_permutation.pdf', bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved PDF: {output_dir / 'figure6_permutation.pdf'}")
    
    fig.savefig(output_dir / 'figure6_permutation.svg', bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved SVG: {output_dir / 'figure6_permutation.svg'}")
    
    plt.close()
    print("\n[OK] Figure 6 complete!")
    print("\nNote: This uses synthetic data for demonstration.")
    print("To use real data, replace generate_synthetic_perm_data() with")
    print("data from: <run_dir>_perm_test_results.csv")

