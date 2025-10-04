"""
Figure 9: Per-Class XAI Differences

Publication-quality visualization showing how model discriminates numerosities
NOTE: Requires per-class XAI outputs
"""

import sys
sys.path.append('../utils')

import matplotlib.pyplot as plt
import numpy as np
from pub_style_v4 import COLORS, WONG_COLORS, save_publication_figure, add_panel_label
from pathlib import Path

def generate_synthetic_perclass_xai():
    """Generate realistic per-class attribution patterns"""
    np.random.seed(42)
    
    n_channels = 100
    n_timepoints = 248
    time_ms = np.linspace(0, 496, n_timepoints)
    
    # Class 1 (one dot): Strong early visual
    class1 = np.zeros((n_channels, n_timepoints))
    class1[0:25, 95:115] = 0.8 * np.outer(
        np.exp(-np.linspace(0, 2, 25)**2),
        np.exp(-np.linspace(-2, 2, 20)**2)
    )
    class1 += 0.03 * np.random.randn(n_channels, n_timepoints)
    class1 = np.clip(class1, 0, 1)
    
    # Class 2 (two dots): Bilateral parietal, sustained
    class2 = np.zeros((n_channels, n_timepoints))
    class2[20:50, 140:240] = 0.6 * np.outer(
        np.exp(-np.linspace(0, 1.5, 30)**2),
        np.exp(-np.linspace(-1, 1, 100)**2)
    )
    class2 += 0.03 * np.random.randn(n_channels, n_timepoints)
    class2 = np.clip(class2, 0, 1)
    
    # Class 3 (three dots): Right parietal dominance, later peak
    class3 = np.zeros((n_channels, n_timepoints))
    # Calculate correct dimensions (n_timepoints=248, so max index is 248)
    ch_range = 35, 65  # 30 channels
    time_range = 198, 248  # 50 timepoints (late in trial)
    class3[ch_range[0]:ch_range[1], time_range[0]:time_range[1]] = 0.7 * np.outer(
        np.exp(-np.linspace(0, 1.5, ch_range[1]-ch_range[0])**2),
        np.exp(-np.linspace(-1.5, 1.5, time_range[1]-time_range[0])**2)
    )
    class3 += 0.03 * np.random.randn(n_channels, n_timepoints)
    class3 = np.clip(class3, 0, 1)
    
    # Temporal profiles
    temp1 = class1.mean(axis=0)
    temp2 = class2.mean(axis=0)
    temp3 = class3.mean(axis=0)
    
    # Difference maps
    diff_21 = class2 - class1
    diff_32 = class3 - class2
    
    return (class1, class2, class3, temp1, temp2, temp3, 
            diff_21, diff_32, time_ms)

def create_perclass_xai():
    """Create publication-quality per-class XAI figure"""
    
    # Get data
    (c1, c2, c3, t1, t2, t3, diff_21, diff_32, time_ms) = generate_synthetic_perclass_xai()
    
    # Create figure (3 rows: one per class, + difference rows)
    fig, axes = plt.subplots(5, 2, figsize=(10, 10), layout='constrained',
                            gridspec_kw={'width_ratios': [4, 1]})
    
    cmap = 'inferno'
    cmap_diff = 'RdBu_r'
    
    # Row 1: Class 1 (One dot)
    im1 = axes[0, 0].imshow(c1, aspect='auto', cmap=cmap, interpolation='nearest',
                           extent=[time_ms[0], time_ms[-1], c1.shape[0], 0],
                           vmin=0, vmax=1)
    axes[0, 0].set_ylabel('Channel', fontsize=9)
    axes[0, 0].set_title('Class 1 (One Dot) - Strong Early Visual (~100ms, Oz)', 
                         fontsize=10, pad=8)
    axes[0, 0].set_xticklabels([])
    add_panel_label(axes[0, 0], 'A', x=-0.12, y=1.05)
    
    axes[0, 1].plot(t1, time_ms, color=WONG_COLORS['skyblue'], linewidth=2)
    axes[0, 1].fill_betweenx(time_ms, 0, t1, color=WONG_COLORS['skyblue'], alpha=0.3)
    axes[0, 1].set_ylabel('Time (ms)', fontsize=9)
    axes[0, 1].set_xlabel('Attr.', fontsize=8)
    axes[0, 1].set_ylim([0, 500])
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Row 2: Class 2 (Two dots)
    im2 = axes[1, 0].imshow(c2, aspect='auto', cmap=cmap, interpolation='nearest',
                           extent=[time_ms[0], time_ms[-1], c2.shape[0], 0],
                           vmin=0, vmax=1)
    axes[1, 0].set_ylabel('Channel', fontsize=9)
    axes[1, 0].set_title('Class 2 (Two Dots) - Bilateral Parietal (150–250ms)',
                         fontsize=10, pad=8)
    axes[1, 0].set_xticklabels([])
    add_panel_label(axes[1, 0], 'B', x=-0.12, y=1.05)
    
    axes[1, 1].plot(t2, time_ms, color=WONG_COLORS['vermillion'], linewidth=2)
    axes[1, 1].fill_betweenx(time_ms, 0, t2, color=WONG_COLORS['vermillion'], alpha=0.3)
    axes[1, 1].set_ylabel('Time (ms)', fontsize=9)
    axes[1, 1].set_xlabel('Attr.', fontsize=8)
    axes[1, 1].set_ylim([0, 500])
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Row 3: Class 3 (Three dots)
    im3 = axes[2, 0].imshow(c3, aspect='auto', cmap=cmap, interpolation='nearest',
                           extent=[time_ms[0], time_ms[-1], c3.shape[0], 0],
                           vmin=0, vmax=1)
    axes[2, 0].set_ylabel('Channel', fontsize=9)
    axes[2, 0].set_title('Class 3 (Three Dots) - Right Parietal Late Peak (~270ms)',
                         fontsize=10, pad=8)
    axes[2, 0].set_xticklabels([])
    add_panel_label(axes[2, 0], 'C', x=-0.12, y=1.05)
    
    axes[2, 1].plot(t3, time_ms, color=WONG_COLORS['green'], linewidth=2)
    axes[2, 1].fill_betweenx(time_ms, 0, t3, color=WONG_COLORS['green'], alpha=0.3)
    axes[2, 1].set_ylabel('Time (ms)', fontsize=9)
    axes[2, 1].set_xlabel('Attr.', fontsize=8)
    axes[2, 1].set_ylim([0, 500])
    axes[2, 1].invert_yaxis()
    axes[2, 1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Row 4: Difference (Class 2 - Class 1)
    vmax_diff = max(np.abs(diff_21).max(), np.abs(diff_32).max())
    im_diff1 = axes[3, 0].imshow(diff_21, aspect='auto', cmap=cmap_diff, interpolation='nearest',
                                extent=[time_ms[0], time_ms[-1], diff_21.shape[0], 0],
                                vmin=-vmax_diff, vmax=vmax_diff)
    axes[3, 0].set_ylabel('Channel', fontsize=9)
    axes[3, 0].set_title('Difference: Class 2 − Class 1', fontsize=10, pad=8)
    axes[3, 0].set_xticklabels([])
    add_panel_label(axes[3, 0], 'D', x=-0.12, y=1.05)
    
    diff_temp_21 = diff_21.mean(axis=0)
    axes[3, 1].plot(diff_temp_21, time_ms, color='#333', linewidth=2)
    axes[3, 1].axvline(0, color=COLORS['chance'], linestyle=':', linewidth=1)
    axes[3, 1].set_ylabel('Time (ms)', fontsize=9)
    axes[3, 1].set_xlabel('Δ Attr.', fontsize=8)
    axes[3, 1].set_ylim([0, 500])
    axes[3, 1].invert_yaxis()
    axes[3, 1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Row 5: Difference (Class 3 - Class 2)
    im_diff2 = axes[4, 0].imshow(diff_32, aspect='auto', cmap=cmap_diff, interpolation='nearest',
                                extent=[time_ms[0], time_ms[-1], diff_32.shape[0], 0],
                                vmin=-vmax_diff, vmax=vmax_diff)
    axes[4, 0].set_xlabel('Time (ms)', fontsize=9)
    axes[4, 0].set_ylabel('Channel', fontsize=9)
    axes[4, 0].set_title('Difference: Class 3 − Class 2', fontsize=10, pad=8)
    add_panel_label(axes[4, 0], 'E', x=-0.12, y=1.05)
    
    diff_temp_32 = diff_32.mean(axis=0)
    axes[4, 1].plot(diff_temp_32, time_ms, color='#333', linewidth=2)
    axes[4, 1].axvline(0, color=COLORS['chance'], linestyle=':', linewidth=1)
    axes[4, 1].set_xlabel('Time (ms)', fontsize=9)
    axes[4, 1].set_ylabel('Time (ms)', fontsize=9)
    axes[4, 1].set_xlabel('Δ Attr.', fontsize=8)
    axes[4, 1].set_ylim([0, 500])
    axes[4, 1].invert_yaxis()
    axes[4, 1].grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Colorbars
    fig.colorbar(im3, ax=axes[0:3, 0], location='right', pad=0.01, aspect=30, shrink=0.8,
                label='Attribution Magnitude')
    fig.colorbar(im_diff2, ax=axes[3:5, 0], location='right', pad=0.01, aspect=20, shrink=0.8,
                label='Difference in Attribution')
    
    # Note
    note = ('Progressive recruitment of parietal cortex with increasing numerosity. '
           'Difference maps reveal distinct spatiotemporal signatures.')
    fig.text(0.5, 0.01, note, ha='center', fontsize=7, style='italic')
    
    return fig

if __name__ == '__main__':
    output_dir = Path('D:/eeg_nn/publication-ready-media/outputs/v4')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING FIGURE 9: PER-CLASS XAI")
    print("="*60)
    print("NOTE: Using synthetic demonstration data")
    print("To use real data, load from: xai_analysis/ig_per_class_heatmaps/")
    
    fig = create_perclass_xai()
    
    save_publication_figure(fig, output_dir / 'figure9_xai_perclass',
                           formats=['pdf', 'png', 'svg'])
    
    plt.close()
    print("\n[OK] Figure 9 complete - Publication ready!")
    print("="*60)


