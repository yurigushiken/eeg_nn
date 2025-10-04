"""
Publication Style V4: Neuroscience Journal Standards
- Conservative muted blues (shades of blue only for flowcharts)
- White backgrounds
- High contrast
- No occlusions
- Clean, professional appearance
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Try to import scienceplots, but don't fail if not available
try:
    import scienceplots
    _has_scienceplots = True
except ImportError:
    _has_scienceplots = False

# ============================================================================
# COLOR PALETTES - CONSERVATIVE BLUES FOR FLOWCHARTS
# ============================================================================

# Muted blues palette (for flowcharts/schematics) - light to dark
BLUES_PALETTE = {
    'very_light': '#E3F2FD',  # Very light blue
    'light': '#BBDEFB',        # Light blue
    'medium_light': '#90CAF9', # Medium-light blue
    'medium': '#64B5F6',       # Medium blue
    'medium_dark': '#42A5F5',  # Medium-dark blue
    'dark': '#2196F3',         # Dark blue
    'very_dark': '#1976D2',    # Very dark blue
    'deepest': '#1565C0',      # Deepest blue
}

# Wong palette (for data plots - colorblind-safe)
WONG_COLORS_LIST = [
    '#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'
]

# Also provide as named dict for easier access
WONG_COLORS = {
    'black': '#000000',
    'orange': '#E69F00',
    'skyblue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'purple': '#CC79A7'
}

# Semantic color mapping for figures
COLORS = {
    # General
    'background': '#FFFFFF',
    'text': '#333333',
    'grid': '#CCCCCC',
    'border': '#333333',

    # Pipeline stages (BLUES ONLY - conservative)
    'data': BLUES_PALETTE['medium_light'],       # Light blue for data
    'preprocessing': BLUES_PALETTE['medium'],    # Medium blue for preprocessing
    'finalization': BLUES_PALETTE['medium_dark'], # Medium-dark for finalization
    'task_split': BLUES_PALETTE['light'],        # Light for task split
    'optimization': BLUES_PALETTE['dark'],       # Dark for optimization
    'evaluation': BLUES_PALETTE['medium_dark'],  # Medium-dark for evaluation
    'analysis': BLUES_PALETTE['very_dark'],      # Very dark for analysis
    'results': BLUES_PALETTE['deepest'],         # Deepest for results

    # CV splits (muted pastels)
    'train': '#A8E6CF',           # Light green
    'validation': '#FFE7A0',      # Light yellow
    'test': '#FFB3BA',            # Light red

    # Performance/Stats (Wong colors for data plots)
    'chance': '#808080',          # Gray for chance line
    'above_chance': WONG_COLORS_LIST[3],    # Green for significant
    'not_significant': WONG_COLORS_LIST[6], # Vermillion for non-significant
    'observed': WONG_COLORS_LIST[6],    # Vermillion for observed
    'null': WONG_COLORS_LIST[2],        # Sky Blue for null

    # Classes (Wong for consistency)
    'class1': WONG_COLORS_LIST[2],
    'class2': WONG_COLORS_LIST[6],
    'class3': WONG_COLORS_LIST[3],
    'class4': WONG_COLORS_LIST[1],
    'class5': WONG_COLORS_LIST[5],
    'class6': WONG_COLORS_LIST[7],
}

# Sequential Blues for heatmaps
SEQUENTIAL_BLUES = plt.cm.Blues(np.linspace(0.2, 1, 6))

# ============================================================================
# MATPLOTLIB STYLE SETTINGS
# ============================================================================

def set_publication_style():
    """Applies neuroscience publication-ready matplotlib style."""
    if _has_scienceplots:
        plt.style.use(['science', 'grid'])
    else:
        # Manual grid style if scienceplots not available
        mpl.rcParams['axes.grid'] = True
        mpl.rcParams['grid.alpha'] = 0.15
        mpl.rcParams['grid.linestyle'] = '--'
        mpl.rcParams['grid.linewidth'] = 0.5

    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=WONG_COLORS_LIST)

    mpl.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'font.size': 9,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.titleweight': 'regular',
        'axes.labelweight': 'regular',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,  # Reduced padding
        'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.04,
        'figure.constrained_layout.w_pad': 0.04,
        'grid.alpha': 0.15,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
    })

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_publication_figure(fig, filename_base, formats=['pdf', 'png', 'svg']):
    """Saves figure in publication-ready formats."""
    output_dir = filename_base.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        full_path = filename_base.with_suffix(f'.{fmt}')
        try:
            fig.savefig(full_path, format=fmt, bbox_inches='tight', pad_inches=0.05)
            dpi_info = f" ({mpl.rcParams['savefig.dpi']} DPI)" if fmt in ['png'] else ""
            vector_info = " (vector, fonts embedded)" if fmt in ['pdf', 'svg'] else ""
            # Use absolute path or just filename to avoid relative_to issues
            try:
                rel_path = full_path.relative_to(Path.cwd())
                print(f"[OK] Saved: {rel_path}{dpi_info}{vector_info}")
            except ValueError:
                print(f"[OK] Saved: {full_path.name}{dpi_info}{vector_info}")
        except Exception as e:
            print(f"[ERROR] Could not save {full_path}: {e}")

def add_panel_label(ax, label, x=-0.1, y=1.05, fontsize=10, weight='bold', **kwargs):
    """Adds panel label (e.g., 'A', 'B') to axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, weight=weight, va='top', ha='left', **kwargs)

def move_legend_outside(ax, loc='upper left', bbox_to_anchor=(1.02, 1), **kwargs):
    """Moves legend outside plotting area."""
    ax.legend(loc=loc, bbox_to_anchor=bbox_to_anchor, frameon=False, **kwargs)

def increase_bottom_margin(fig, margin=0.12):
    """Increases bottom margin for footnotes."""
    try:
        fig.subplots_adjust(bottom=margin)
    except:
        # If using constrained_layout, this will fail - that's OK
        pass

def get_figure_size(width_inches=7.0, height_inches=None, aspect_ratio=0.75):
    """Calculate figure size for publications."""
    if height_inches is None:
        height_inches = width_inches * aspect_ratio
    return (width_inches, height_inches)

def add_subtle_grid(ax, axis='both'):
    """Add subtle grid to axes."""
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5, axis=axis)

def create_flowchart_box(ax, xy, width, height, text, facecolor='lightblue',
                         edgecolor='#333', linewidth=1.0, fontsize=8, **kwargs):
    """Creates a flowchart box with text."""
    from matplotlib.patches import FancyBboxPatch
    
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle='round,pad=0.05',
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs
    )
    ax.add_patch(box)
    
    # Add text centered in box
    text_x = xy[0] + width / 2
    text_y = xy[1] + height / 2
    ax.text(text_x, text_y, text, ha='center', va='center',
            fontsize=fontsize, color='#333', wrap=True)

def create_arrow(ax, start, end, color='#555', linewidth=1.5, **kwargs):
    """Creates a flowchart arrow."""
    from matplotlib.patches import FancyArrowPatch
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='-|>',
        mutation_scale=12,
        linewidth=linewidth,
        color=color,
        **kwargs
    )
    ax.add_patch(arrow)

# ============================================================================
# INITIALIZE ON IMPORT
# ============================================================================
set_publication_style()

print("[pub_style_v4] Neuroscience publication style loaded")
print(f"[pub_style_v4] Palette: Muted blues (flowcharts), Wong (data plots)")
print(f"[pub_style_v4] Font: {mpl.rcParams['font.sans-serif'][0]}, {mpl.rcParams['font.size']}pt")
print(f"[pub_style_v4] Save DPI: {mpl.rcParams['savefig.dpi']}")

