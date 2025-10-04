"""
Publication-Quality Style Settings
Implements PI feedback: colorblind-safe palettes, flat colors, consistent typography
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# WONG COLORBLIND-SAFE PALETTE (Nature Methods)
# ============================================================================
WONG_COLORS = {
    'black': '#000000',
    'orange': '#E69F00',
    'skyblue': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7'
}

# Semantic color assignments for our project
COLORS = {
    # Pipeline stages
    'data': WONG_COLORS['skyblue'],          # Data acquisition/finalization
    'preprocessing': WONG_COLORS['reddish_purple'],  # HAPPE preprocessing
    'optimization': WONG_COLORS['vermillion'],       # Optuna stages
    'evaluation': WONG_COLORS['green'],             # Final evaluation
    'statistics': WONG_COLORS['orange'],            # Stats/XAI
    
    # Cross-validation
    'train': '#B3E0B3',        # Light green (desaturated from Wong green)
    'validation': '#FFE6A0',   # Light yellow (desaturated from Wong yellow)
    'test': '#FFB3B3',         # Light red (not Wong, but universally understood)
    
    # Emphasis
    'observed': WONG_COLORS['vermillion'],  # Observed performance
    'null': WONG_COLORS['blue'],            # Null distribution
    'chance': '#666666',                     # Chance level (neutral gray)
}

# ============================================================================
# PUBLICATION SETTINGS
# ============================================================================
def set_publication_style():
    """
    Apply publication-quality matplotlib settings
    Following Nature/Science/PNAS guidelines
    """
    plt.rcdefaults()  # Reset first
    
    mpl.rcParams.update({
        # Figure settings
        'figure.dpi': 150,                     # Screen display
        'figure.autolayout': False,            # Use constrained_layout per-figure
        'figure.facecolor': 'white',
        
        # Saving settings
        'savefig.dpi': 600,                    # Print quality (Nature: 600-1200 for line art)
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,                    # Embed TrueType fonts
        'ps.fonttype': 42,
        
        # Font settings (PI: 8-9pt at final size)
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'font.size': 9,
        'axes.titlesize': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        
        # Axis settings (PI: remove unnecessary spines)
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelweight': 'regular',         # Not bold
        'axes.titleweight': 'regular',
        'axes.spines.top': False,              # Cleaner look
        'axes.spines.right': False,
        
        # Grid (subtle, if used)
        'axes.grid': False,                    # Off by default
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Legend
        'legend.frameon': False,               # No box around legend
        'legend.loc': 'best',
        
        # Lines & patches
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.8,
        'patch.edgecolor': '#333333',
        
        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_figure_size(layout='single_column'):
    """
    Standard figure sizes for journals
    
    Parameters
    ----------
    layout : str
        'single_column': 3.5" width (Nature single column)
        'double_column': 7" width (Nature double column)
        'full_page': 8" width (full page)
        'square': square aspect ratio
    """
    sizes = {
        'single_column': (3.5, 2.6),   # inches
        'double_column': (7.0, 5.0),
        'full_page': (8.0, 10.0),
        'square': (5.0, 5.0),
    }
    return sizes.get(layout, (6.0, 4.0))

def save_publication_figure(fig, filepath, formats=['pdf', 'png']):
    """
    Save figure in publication-ready formats
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : str or Path
        Path without extension
    formats : list
        Output formats (default: ['pdf', 'png'])
    """
    from pathlib import Path
    filepath = Path(filepath)
    
    for fmt in formats:
        if fmt == 'pdf':
            fig.savefig(f"{filepath}.pdf", format='pdf', bbox_inches='tight')
            print(f"[OK] Saved: {filepath}.pdf (vector)")
        elif fmt == 'png':
            fig.savefig(f"{filepath}.png", format='png', dpi=600, bbox_inches='tight')
            print(f"[OK] Saved: {filepath}.png (600 DPI)")
        elif fmt == 'svg':
            fig.savefig(f"{filepath}.svg", format='svg', bbox_inches='tight')
            print(f"[OK] Saved: {filepath}.svg (vector)")

def add_panel_label(ax, label, x=-0.05, y=1.05):
    """
    Add panel label (A, B, C, etc.) to subplot
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    label : str
        Panel label (e.g., 'A', 'B')
    x, y : float
        Position in axes coordinates
    """
    ax.text(x, y, label, transform=ax.transAxes,
           fontsize=11, fontweight='bold', va='top', ha='right')

# ============================================================================
# COLOR-SPECIFIC UTILITIES
# ============================================================================
def get_colorblind_palette(n=8):
    """Get n colors from Wong palette"""
    wong_list = list(WONG_COLORS.values())
    return wong_list[:n]

def lighten_color(color, amount=0.3):
    """
    Lighten a color by blending with white
    Useful for creating flat fill colors from base palette
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

# ============================================================================
# BOX/ARROW HELPERS FOR FLOWCHARTS
# ============================================================================
def create_flowchart_box(ax, xy, width, height, text, 
                        facecolor=COLORS['data'], 
                        edgecolor='#333333',
                        linewidth=1.2,
                        rounding=0.05,
                        fontsize=8,
                        zorder=2):
    """
    Create a publication-quality box for flowcharts
    
    Follows PI feedback:
    - Flat colors (no gradients)
    - Consistent corner radius
    - Consistent stroke width
    - No shadows
    """
    from matplotlib.patches import FancyBboxPatch
    
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle=f'round,pad=0.02,rounding_size={rounding}',
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
        clip_on=False
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(
        xy[0] + width/2, xy[1] + height/2,
        text,
        ha='center', va='center',
        fontsize=fontsize,
        wrap=True,
        zorder=zorder+1
    )
    return box

def create_arrow(ax, start, end, 
                linewidth=1.1, 
                color='#555555',
                arrowstyle='-|>',
                mutation_scale=10,
                zorder=1):
    """
    Create consistent arrow for flowcharts
    """
    from matplotlib.patches import FancyArrowPatch
    
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        zorder=zorder,
        clip_on=False
    )
    ax.add_patch(arrow)
    return arrow

# ============================================================================
# INITIALIZE ON IMPORT
# ============================================================================
set_publication_style()

print("[pub_style] Publication style loaded")
print(f"[pub_style] Wong colorblind-safe palette active")
print(f"[pub_style] Font: {mpl.rcParams['font.sans-serif'][0]}, {mpl.rcParams['font.size']}pt")
print(f"[pub_style] Save DPI: {mpl.rcParams['savefig.dpi']}")

