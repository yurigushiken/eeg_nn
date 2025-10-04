"""
Publication-Quality Style Settings V3 - PI Feedback Implemented
Locked palettes (Wong + Tol), professional spacing, legend management
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# LOCKED COLOR PALETTES (Colorblind-Safe)
# ============================================================================

# Wong Palette (Nature Methods, 2011) - PRIMARY PALETTE
# https://www.nature.com/articles/nmeth.1618
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

# Wong palette as list (for cyclers)
WONG_LIST = [
    '#000000',  # black
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#F0E442',  # yellow
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7'   # reddish purple
]

# Paul Tol Palettes - SECONDARY PALETTES
# https://personal.sron.nl/~pault/
TOL_MUTED = [
    '#332288',  # indigo
    '#88CCEE',  # cyan
    '#44AA99',  # teal
    '#117733',  # green
    '#999933',  # olive
    '#DDCC77',  # sand
    '#CC6677',  # rose
    '#882255',  # wine
    '#AA4499'   # purple
]

TOL_BRIGHT = [
    '#4477AA',  # blue
    '#EE6677',  # red
    '#228833',  # green
    '#CCBB44',  # yellow
    '#66CCEE',  # cyan
    '#AA3377',  # purple
    '#BBBBBB'   # grey
]

# Sequential colormap (for heatmaps) - Blues, colorblind-safe
SEQUENTIAL_BLUES = 'Blues'  # matplotlib built-in, colorblind-safe

# Diverging colormap (for difference maps) - RdBu, colorblind-safe
DIVERGING_RDBU = 'RdBu_r'  # matplotlib built-in, symmetric, colorblind-safe

# ============================================================================
# SEMANTIC COLOR ASSIGNMENTS (LOCKED)
# ============================================================================
COLORS = {
    # Pipeline stages (Wong palette)
    'data': WONG_COLORS['skyblue'],          # #56B4E9
    'preprocessing': WONG_COLORS['reddish_purple'],  # #CC79A7
    'optimization': WONG_COLORS['vermillion'],  # #D55E00
    'evaluation': WONG_COLORS['green'],      # #009E73
    'statistics': WONG_COLORS['orange'],     # #E69F00
    
    # Cross-validation (desaturated Wong)
    'train': '#B3E0B3',        # Light green
    'validation': '#FFE6A0',   # Light yellow
    'test': '#FFB3B3',         # Light red
    
    # Emphasis
    'observed': WONG_COLORS['vermillion'],  # #D55E00
    'null': WONG_COLORS['blue'],            # #0072B2
    'chance': '#666666',                     # Neutral gray
    
    # Per-class colors (from Wong, consistent across all figures)
    'class1': WONG_COLORS['skyblue'],       # #56B4E9
    'class2': WONG_COLORS['vermillion'],    # #D55E00
    'class3': WONG_COLORS['green'],         # #009E73
}

# ============================================================================
# PUBLICATION SETTINGS (PI Feedback Implemented)
# ============================================================================
def set_publication_style():
    """
    Apply publication-quality matplotlib settings
    Following Nature/Science/PNAS guidelines + PI feedback
    """
    plt.rcdefaults()  # Reset first
    
    mpl.rcParams.update({
        # Figure settings
        'figure.dpi': 150,                     # Screen display
        'figure.autolayout': False,            # Use constrained_layout per-figure
        'figure.facecolor': 'white',
        
        # Saving settings (PI: vector PDF primary, 600 DPI raster)
        'savefig.dpi': 600,                    # Nature: 600-1200 for line art
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,            # PI: Minimal padding
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,                    # PI: Embed TrueType fonts
        'ps.fonttype': 42,
        
        # Font settings (PI: 9pt body, 8pt annotations at print size)
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'font.size': 9,                        # Body text
        'axes.titlesize': 9,                   # Panel titles (use sparingly)
        'axes.labelsize': 9,                   # Axis labels
        'xtick.labelsize': 8,                  # Tick labels
        'ytick.labelsize': 8,
        'legend.fontsize': 8,                  # Legend
        
        # Axis settings (PI: remove unnecessary spines)
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'axes.labelweight': 'regular',         # PI: No bold mid-sentence
        'axes.titleweight': 'regular',
        'axes.spines.top': False,              # PI: Cleaner look
        'axes.spines.right': False,
        
        # Grid (PI: subtle if used)
        'axes.grid': False,                    # Off by default
        'grid.alpha': 0.15,                    # PI: Very subtle
        'grid.linewidth': 0.4,
        'grid.linestyle': '--',
        
        # Legend (PI: outside plots, no frame)
        'legend.frameon': False,               # No box
        'legend.loc': 'best',
        'legend.fancybox': False,              # Simple corners
        
        # Lines & patches
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.8,
        'patch.edgecolor': '#333333',
        
        # Ticks
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        
        # Layout (PI: constrained_layout for consistent spacing)
        'figure.constrained_layout.use': False,  # Set per-figure
    })
    
    # Set Wong palette as default cycler
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=WONG_LIST)

# ============================================================================
# HELPER FUNCTIONS (PI Feedback Implemented)
# ============================================================================

def get_figure_size(layout='double_column'):
    """
    Standard figure sizes for journals
    
    Parameters
    ----------
    layout : str
        'single_column': 89mm (3.5") - Nature single column
        'double_column': 183mm (7.2") - Nature double column
        'full_page': 8" width
        'square': square aspect ratio
    """
    sizes = {
        'single_column': (3.5, 2.6),   # 89mm
        'double_column': (7.2, 5.0),   # 183mm
        'full_page': (8.0, 10.0),
        'square': (5.0, 5.0),
    }
    return sizes.get(layout, (7.0, 5.0))

def save_publication_figure(fig, filepath, formats=['pdf', 'png']):
    """
    Save figure in publication-ready formats
    PI feedback: Vector PDF primary, 600 DPI PNG backup
    """
    from pathlib import Path
    filepath = Path(filepath)
    
    for fmt in formats:
        if fmt == 'pdf':
            fig.savefig(f"{filepath}.pdf", format='pdf')
            print(f"[OK] Saved: {filepath}.pdf (vector, fonts embedded)")
        elif fmt == 'png':
            fig.savefig(f"{filepath}.png", format='png', dpi=600)
            print(f"[OK] Saved: {filepath}.png (600 DPI)")
        elif fmt == 'svg':
            fig.savefig(f"{filepath}.svg", format='svg')
            print(f"[OK] Saved: {filepath}.svg (vector, editable)")

def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=11):
    """
    Add panel label (A, B, C, etc.) to subplot
    PI feedback: Keep modest, not >12pt
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    label : str
        Panel label (e.g., 'A', 'B')
    x, y : float
        Position in axes coordinates
    fontsize : int
        Label font size (PI: 9-11pt, not >12)
    """
    ax.text(x, y, label, transform=ax.transAxes,
           fontsize=fontsize, fontweight='bold', va='top', ha='right')

def move_legend_outside(ax, loc='upper left', bbox=(1.02, 1)):
    """
    Move legend outside plot area (PI feedback)
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    loc : str
        Legend location
    bbox : tuple
        bbox_to_anchor position
    """
    legend = ax.legend(loc=loc, bbox_to_anchor=bbox, borderaxespad=0,
                      frameon=False, fancybox=False)
    return legend

def add_subtle_grid(ax, axis='y', alpha=0.15):
    """
    Add subtle grid (PI feedback)
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
    axis : str
        'x', 'y', or 'both'
    alpha : float
        Grid transparency (PI: ~0.15)
    """
    ax.grid(True, axis=axis, alpha=alpha, linestyle='--', linewidth=0.4, zorder=0)

def increase_bottom_margin(fig, margin=0.15):
    """
    Increase bottom margin for footnotes (PI feedback)
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
    margin : float
        Bottom margin as fraction of figure height
    """
    fig.subplots_adjust(bottom=margin)

def add_caption_text(fig, text, y=0.01, fontsize=7):
    """
    Add caption-style text at bottom of figure (PI feedback: move footnotes to captions)
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
    text : str
        Caption text
    y : float
        Vertical position (figure coordinates)
    fontsize : int
        Font size
    """
    fig.text(0.5, y, text, ha='center', va='bottom', fontsize=fontsize, 
            style='italic', wrap=True)

# ============================================================================
# BOX/ARROW HELPERS FOR FLOWCHARTS
# ============================================================================
def create_flowchart_box(ax, xy, width, height, text, 
                        facecolor=COLORS['data'], 
                        edgecolor='#333333',
                        linewidth=1.2,
                        rounding=0.015,
                        fontsize=8,
                        zorder=2):
    """
    Create publication-quality box for flowcharts (PI feedback: flat colors, consistent spacing)
    """
    from matplotlib.patches import FancyBboxPatch
    
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle=f'round,pad=0.015,rounding_size={rounding}',  # PI: consistent rounding
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
    Create consistent arrow for flowcharts (PI feedback: same size and stroke)
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
# CONFUSION MATRIX HELPER (Using sklearn)
# ============================================================================
def plot_confusion_matrix_enhanced(cm, class_names, ax, cmap=SEQUENTIAL_BLUES,
                                   normalize='true', title=None):
    """
    Plot confusion matrix using sklearn-style with enhancements
    
    Parameters
    ----------
    cm : array-like
        Confusion matrix
    class_names : list
        Class labels
    ax : matplotlib.axes.Axes
    cmap : str
        Colormap name (default: Blues, colorblind-safe)
    normalize : str
        'true' (row-wise), 'pred' (column-wise), 'all', or None
    title : str
        Panel title (keep concise per PI feedback)
    """
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    # Normalize if requested
    if normalize == 'true':
        cm_display = 100 * cm / cm.sum(axis=1, keepdims=True)
        fmt = '.0f'
        label = 'Percentage of Trials (%)'
    elif normalize == 'pred':
        cm_display = 100 * cm / cm.sum(axis=0, keepdims=True)
        fmt = '.0f'
        label = 'Percentage (%)'
    else:
        cm_display = cm
        fmt = 'd'
        label = 'Count'
    
    # Plot
    im = ax.imshow(cm_display, cmap=cmap, aspect='auto', vmin=0, 
                  vmax=100 if normalize else None)
    
    # Add values to cells
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_display[i, j]
            text_color = 'white' if value > 50 else 'black'
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i, f'{value:{fmt}}', ha='center', va='center',
                   fontsize=8, color=text_color, weight=weight)
    
    # Labels
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)
    
    if title:
        ax.set_title(title, fontsize=10, pad=8)
    
    return im

# ============================================================================
# INITIALIZE ON IMPORT
# ============================================================================
set_publication_style()

print("[pub_style_v3] Publication style loaded (PI feedback implemented)")
print(f"[pub_style_v3] Primary palette: Wong (8 colors, colorblind-safe)")
print(f"[pub_style_v3] Secondary palette: Tol Muted (9 colors)")
print(f"[pub_style_v3] Font: {mpl.rcParams['font.sans-serif'][0]}, {mpl.rcParams['font.size']}pt")
print(f"[pub_style_v3] Save DPI: {mpl.rcParams['savefig.dpi']} (vector PDF primary)")
print("[pub_style_v3] Legends: Outside plots, no frame")
print("[pub_style_v3] Grid: Subtle (alpha=0.15), off by default")

