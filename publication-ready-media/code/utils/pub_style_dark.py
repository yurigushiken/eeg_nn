"""
Publication-Quality DARK Style Settings - Nord-Inspired
Muted dark blue theme for presentations and dark-mode publications
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# NORD PALETTE (Dark, Muted, Blue-Focused)
# ============================================================================
# Based on Nord color scheme - popular in scientific/technical presentations
# https://www.nordtheme.com/

# Polar Night (dark backgrounds)
NORD_POLAR = {
    'darkest': '#2E3440',      # Background
    'dark': '#3B4252',          # Elevated background
    'medium': '#434C5E',        # Secondary background
    'light': '#4C566A'          # Tertiary background
}

# Snow Storm (light text)
NORD_SNOW = {
    'light': '#D8DEE9',         # Primary text
    'lighter': '#E5E9F0',       # Secondary text
    'lightest': '#ECEFF4'       # Brightest text
}

# Frost (blue tones - PRIMARY DATA COLORS)
NORD_FROST = {
    'cyan': '#8FBCBB',          # Teal-cyan
    'bright_cyan': '#88C0D0',   # Bright cyan
    'blue': '#81A1C1',          # Mid blue
    'dark_blue': '#5E81AC'      # Dark blue
}

# Aurora (accent colors for differentiation)
NORD_AURORA = {
    'red': '#BF616A',           # Muted red
    'orange': '#D08770',        # Muted orange
    'yellow': '#EBCB8B',        # Muted yellow
    'green': '#A3BE8C',         # Muted green
    'purple': '#B48EAD'         # Muted purple
}

# Combined palette as list (for cyclers)
NORD_PALETTE = [
    NORD_FROST['bright_cyan'],  # #88C0D0 - Primary (cyan)
    NORD_AURORA['green'],       # #A3BE8C - Secondary (green)
    NORD_AURORA['purple'],      # #B48EAD - Tertiary (purple)
    NORD_AURORA['orange'],      # #D08770 - Quaternary (orange)
    NORD_FROST['blue'],         # #81A1C1 - Quinary (blue)
    NORD_AURORA['yellow'],      # #EBCB8B - Senary (yellow)
    NORD_FROST['dark_blue'],    # #5E81AC - Septenary (dark blue)
    NORD_AURORA['red']          # #BF616A - Octonary (red)
]

# ============================================================================
# SEMANTIC COLOR ASSIGNMENTS (DARK THEME)
# ============================================================================
DARK_COLORS = {
    # Pipeline stages
    'data': NORD_FROST['bright_cyan'],        # #88C0D0
    'preprocessing': NORD_AURORA['purple'],   # #B48EAD
    'optimization': NORD_AURORA['orange'],    # #D08770
    'evaluation': NORD_AURORA['green'],       # #A3BE8C
    'statistics': NORD_FROST['blue'],         # #81A1C1
    
    # Cross-validation
    'train': NORD_AURORA['green'],            # #A3BE8C
    'validation': NORD_AURORA['yellow'],      # #EBCB8B
    'test': NORD_AURORA['red'],               # #BF616A
    
    # Emphasis
    'observed': NORD_AURORA['orange'],        # #D08770
    'null': NORD_FROST['blue'],               # #81A1C1
    'chance': NORD_SNOW['light'],             # #D8DEE9 (light gray)
    
    # Per-class colors (consistent)
    'class1': NORD_FROST['bright_cyan'],      # #88C0D0
    'class2': NORD_AURORA['orange'],          # #D08770
    'class3': NORD_AURORA['green'],           # #A3BE8C
    
    # Text
    'text_primary': NORD_SNOW['lightest'],    # #ECEFF4
    'text_secondary': NORD_SNOW['light'],     # #D8DEE9
    'text_tertiary': NORD_POLAR['light'],     # #4C566A
}

# ============================================================================
# DARK PUBLICATION SETTINGS
# ============================================================================
def set_dark_publication_style():
    """
    Apply dark publication-quality matplotlib settings
    Nord-inspired dark blue muted theme
    """
    plt.rcdefaults()  # Reset first
    
    mpl.rcParams.update({
        # Figure settings (dark background)
        'figure.dpi': 150,
        'figure.autolayout': False,
        'figure.facecolor': NORD_POLAR['darkest'],    # #2E3440
        'figure.edgecolor': NORD_POLAR['darkest'],
        
        # Saving settings
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,                    # Extra padding for dark theme
        'savefig.facecolor': NORD_POLAR['darkest'],
        'savefig.edgecolor': NORD_POLAR['darkest'],
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        
        # Font settings
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'font.size': 10,                               # Slightly larger for dark theme
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Axes settings (dark background, light foreground)
        'axes.facecolor': NORD_POLAR['dark'],          # #3B4252
        'axes.edgecolor': NORD_SNOW['light'],          # #D8DEE9
        'axes.linewidth': 1.0,
        'axes.labelcolor': NORD_SNOW['lightest'],      # #ECEFF4
        'axes.titlecolor': NORD_SNOW['lightest'],
        'axes.labelweight': 'regular',
        'axes.titleweight': 'regular',
        'axes.spines.top': False,
        'axes.spines.right': False,
        
        # Grid (subtle on dark)
        'axes.grid': False,
        'grid.alpha': 0.2,                             # Slightly more visible on dark
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        'grid.color': NORD_POLAR['light'],             # #4C566A
        
        # Legend (dark)
        'legend.frameon': False,
        'legend.loc': 'best',
        'legend.fancybox': False,
        'legend.facecolor': NORD_POLAR['medium'],      # #434C5E
        'legend.edgecolor': NORD_SNOW['light'],
        
        # Lines & patches
        'lines.linewidth': 1.5,                        # Slightly thicker for visibility
        'patch.linewidth': 1.0,
        'patch.edgecolor': NORD_SNOW['light'],
        
        # Ticks (light colored)
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4.0,
        'ytick.major.size': 4.0,
        'xtick.color': NORD_SNOW['light'],             # #D8DEE9
        'ytick.color': NORD_SNOW['light'],
        
        # Text color (light)
        'text.color': NORD_SNOW['lightest'],           # #ECEFF4
    })
    
    # Set Nord palette as default cycler
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=NORD_PALETTE)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def add_panel_label(ax, label, x=-0.15, y=1.05, fontsize=11):
    """Add panel label (A, B, C) to subplot - dark theme version"""
    ax.text(x, y, label, transform=ax.transAxes,
           fontsize=fontsize, fontweight='bold', va='top', ha='right',
           color=NORD_SNOW['lightest'])

def increase_bottom_margin(fig, margin=0.15):
    """Increase bottom margin for footnotes"""
    try:
        fig.subplots_adjust(bottom=margin)
    except:
        pass  # Ignore if layout incompatible

def add_subtle_grid(ax, axis='y', alpha=0.2):
    """Add subtle grid for dark theme"""
    ax.grid(True, axis=axis, alpha=alpha, linestyle='--', linewidth=0.5, zorder=0)

def save_dark_figure(fig, filepath, formats=['pdf', 'png', 'svg']):
    """
    Save dark-themed figure in publication-ready formats
    """
    from pathlib import Path
    filepath = Path(filepath)
    
    for fmt in formats:
        if fmt == 'pdf':
            fig.savefig(f"{filepath}.pdf", format='pdf', 
                       facecolor=NORD_POLAR['darkest'], edgecolor='none')
            print(f"[OK] Saved: {filepath}.pdf (dark vector)")
        elif fmt == 'png':
            fig.savefig(f"{filepath}.png", format='png', dpi=600,
                       facecolor=NORD_POLAR['darkest'], edgecolor='none')
            print(f"[OK] Saved: {filepath}.png (dark 600 DPI)")
        elif fmt == 'svg':
            fig.savefig(f"{filepath}.svg", format='svg',
                       facecolor=NORD_POLAR['darkest'], edgecolor='none')
            print(f"[OK] Saved: {filepath}.svg (dark vector editable)")

def create_dark_flowchart_box(ax, xy, width, height, text,
                              facecolor=DARK_COLORS['data'],
                              edgecolor=NORD_SNOW['light'],
                              textcolor=NORD_SNOW['lightest'],
                              linewidth=1.5,
                              rounding=0.02,
                              fontsize=9,
                              zorder=2):
    """
    Create dark-themed box for flowcharts
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
        color=textcolor,
        wrap=True,
        zorder=zorder+1
    )
    return box

def create_dark_arrow(ax, start, end,
                     linewidth=1.5,
                     color=NORD_SNOW['light'],
                     arrowstyle='-|>',
                     mutation_scale=12,
                     zorder=1):
    """
    Create dark-themed arrow for flowcharts
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
set_dark_publication_style()

print("[pub_style_dark] Dark publication style loaded (Nord-inspired)")
print(f"[pub_style_dark] Theme: Muted dark blue (Nord palette)")
print(f"[pub_style_dark] Background: {NORD_POLAR['darkest']} (dark)")
print(f"[pub_style_dark] Text: {NORD_SNOW['lightest']} (light)")
print(f"[pub_style_dark] Primary palette: 8 colors, muted, accessible")
print(f"[pub_style_dark] Save DPI: {mpl.rcParams['savefig.dpi']} (vector primary)")

