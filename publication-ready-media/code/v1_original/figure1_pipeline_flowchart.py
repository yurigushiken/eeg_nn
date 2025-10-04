"""
Figure 1: Overall Research Pipeline Flowchart
Publication-ready visualization of end-to-end workflow

Requirements: matplotlib, numpy
Output: High-resolution PNG + vector PDF
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Set publication style
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300

def create_pipeline_flowchart():
    """Create a comprehensive pipeline flowchart"""
    
    fig, ax = plt.subplots(figsize=(10, 14))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # Color scheme (professional academic)
    color_data = '#4A90E2'  # Blue
    color_preproc = '#7B68EE'  # Medium purple
    color_optuna = '#E27B4A'  # Orange
    color_eval = '#50C878'  # Emerald
    color_stats = '#FFB74D'  # Amber
    
    # Helper function to create boxes
    def add_box(x, y, width, height, text, color, fontsize=10, bold=False):
        """Add a fancy box with text"""
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            edgecolor='black',
            facecolor=color,
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)
        
        # Add text
        weight = 'bold' if bold else 'normal'
        ax.text(
            x + width/2, y + height/2,
            text,
            ha='center', va='center',
            fontsize=fontsize,
            weight=weight,
            wrap=True
        )
    
    # Helper function for arrows
    def add_arrow(x1, y1, x2, y2, label=''):
        """Add arrow between boxes"""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color='black',
            linewidth=2,
            zorder=1
        )
        ax.add_patch(arrow)
        
        if label:
            mid_x, mid_y = (x1 + x2)/2, (y1 + y2)/2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8, style='italic')
    
    # Stage 1: Data Acquisition
    y_pos = 19
    add_box(1, y_pos, 8, 0.8, 
            'RAW DATA ACQUISITION\n24 subjects × ~360 trials/subject\n128-channel EEI EEG + Behavioral responses',
            color_data, fontsize=9, bold=True)
    
    # Arrow down
    add_arrow(5, y_pos, 5, y_pos - 0.8)
    
    # Stage 2: HAPPE Preprocessing
    y_pos -= 1.5
    add_box(1, y_pos, 8, 1.2,
            'PREPROCESSING (HAPPE)\n• Bandpass filter: 1.5-40 Hz\n• ICA artifact rejection\n• Bad channel interpolation\n• Epoch: -200 to +800 ms',
            color_preproc, fontsize=9, bold=True)
    
    add_arrow(5, y_pos, 5, y_pos - 1.0)
    
    # Stage 3: Data Finalization
    y_pos -= 2.2
    add_box(1, y_pos, 8, 1.0,
            'DATA FINALIZATION & QC\n• Behavioral alignment validation\n• Min trials per class filter\n• Channel intersection (100 channels)',
            color_data, fontsize=9, bold=True)
    
    add_arrow(5, y_pos, 5, y_pos - 1.0)
    
    # Stage 4: Task Split
    y_pos -= 1.8
    add_box(0.5, y_pos, 4, 0.7, 
            'CARDINALITY 1-3\n(n=3 classes)',
            color_data, fontsize=9, bold=True)
    add_box(5.5, y_pos, 4, 0.7,
            'CARDINALITY 4-6\n(n=3 classes)',
            color_data, fontsize=9, bold=True)
    
    # Arrows converging to Optuna
    add_arrow(2.5, y_pos, 5, y_pos - 0.9)
    add_arrow(7.5, y_pos, 5, y_pos - 0.9)
    
    # Stage 5: Optuna (3 stages side by side)
    y_pos -= 2.2
    add_box(0.3, y_pos + 0.5, 9.4, 0.4,
            '3-STAGE HYPERPARAMETER OPTIMIZATION (Optuna/TPE)',
            color_optuna, fontsize=10, bold=True)
    
    # Stage 1, 2, 3 boxes
    stage_y = y_pos - 0.8
    add_box(0.5, stage_y, 2.8, 1.3,
            'STAGE 1\nArchitecture\n~50 trials\n\n• depth_multiplier\n• filters\n• kernels',
            color_optuna, fontsize=8)
    add_box(3.6, stage_y, 2.8, 1.3,
            'STAGE 2\nLearning\n~50 trials\n\n• learning rate\n• warmup\n• scheduler',
            color_optuna, fontsize=8)
    add_box(6.7, stage_y, 2.8, 1.3,
            'STAGE 3\nAugmentation\n~30 trials\n\n• shift, scale\n• noise, mask',
            color_optuna, fontsize=8)
    
    # Arrows between stages
    add_arrow(3.3, stage_y + 0.65, 3.6, stage_y + 0.65, 'winner →')
    add_arrow(6.4, stage_y + 0.65, 6.7, stage_y + 0.65, 'winner →')
    
    # Objective note
    ax.text(5, stage_y - 0.5, 
            'Objective: inner_mean_min_per_class_f1\n(ensures ALL classes decodable)',
            ha='center', fontsize=8, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))
    
    add_arrow(5, stage_y - 0.7, 5, stage_y - 1.3)
    
    # Stage 6: Final Evaluation
    y_pos -= 5.5
    add_box(1, y_pos, 8, 1.0,
            'FINAL EVALUATION (Multi-seed LOSO-CV)\n• 10 independent seeds\n• Leave-One-Subject-Out\n• Nested 5-fold inner CV\n• Ensemble predictions',
            color_eval, fontsize=9, bold=True)
    
    # Split to two paths
    add_arrow(3, y_pos, 3, y_pos - 1.0)
    add_arrow(7, y_pos, 7, y_pos - 1.0)
    
    # Stage 7: Parallel boxes (Stats and XAI)
    y_pos -= 2.5
    add_box(0.5, y_pos, 4, 1.3,
            'STATISTICAL VALIDATION\n\n• Permutation testing\n  (n=200)\n• Per-subject\n  significance\n• Mixed-effects\n  (GLMM)',
            color_stats, fontsize=9, bold=True)
    add_box(5.5, y_pos, 4, 1.3,
            'EXPLAINABLE AI (XAI)\n\n• Integrated Gradients\n• Channel importance\n• Temporal dynamics\n• Spatiotemporal\n  patterns',
            color_stats, fontsize=9, bold=True)
    
    # Converge to final results
    add_arrow(2.5, y_pos, 5, y_pos - 1.5)
    add_arrow(7.5, y_pos, 5, y_pos - 1.5)
    
    # Stage 8: Results
    y_pos -= 2.5
    add_box(1, y_pos, 8, 1.0,
            'PUBLICATION RESULTS\n• Mean accuracy: 48-52% (chance: 33.3%)\n• All classes decodable (min-F1 > 35%)\n• p < 0.01 (permutation test)\n• Early visual + late parietal patterns',
            '#90EE90', fontsize=9, bold=True)
    
    # Title
    ax.text(5, 20.5, 
            'End-to-End Pipeline: Cardinal Numerosity Decoding from Single-Trial EEG',
            ha='center', fontsize=14, weight='bold')
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    output_dir = Path('../outputs/figure1_pipeline')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig = create_pipeline_flowchart()
    
    # Save as high-res PNG
    fig.savefig(output_dir / 'figure1_pipeline.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved PNG: {output_dir / 'figure1_pipeline.png'}")
    
    # Save as vector PDF
    fig.savefig(output_dir / 'figure1_pipeline.pdf', bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved PDF: {output_dir / 'figure1_pipeline.pdf'}")
    
    # Save as SVG for editing
    fig.savefig(output_dir / 'figure1_pipeline.svg', bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved SVG: {output_dir / 'figure1_pipeline.svg'}")
    
    plt.close()
    print("\n[OK] Figure 1 complete!")

