"""
Utility functions for visualizing channel selection in EEG experiments.

This module provides functions to create publication-quality topomaps showing
which channels are actually used vs. excluded in a given configuration.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mne


def create_channel_usage_topomap(
    montage_path: Path,
    used_channels: list,
    output_path: Path,
    title: str = "Channel Selection",
    dpi: int = 300
):
    """
    Create a professional topomap showing which channels are used.
    
    Args:
        montage_path: Path to .sfp montage file
        used_channels: List of channel names that are actually used
        output_path: Where to save the figure
        title: Figure title
        dpi: Resolution for saved figure
    
    Returns:
        None (saves figure to output_path)
    """
    # Load montage
    montage = mne.channels.read_custom_montage(str(montage_path))
    all_channels = montage.ch_names
    
    # Create an info object with all channels
    info = mne.create_info(
        ch_names=all_channels,
        sfreq=1000.0,  # Dummy sampling rate
        ch_types='eeg'
    )
    info.set_montage(montage)
    
    # Create a mask: 1 for used channels, 0 for excluded
    mask = np.array([ch in used_channels for ch in all_channels])
    
    # Create data: use mask values as "activation"
    data = mask.astype(float)
    
    # Create figure with single panel
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    # Show all channels with used ones highlighted
    
    # Plot with used channels at value 1, excluded at value 0
    # Use fixed sphere for consistent, standardized view across all trials
    # The .sfp file represents realistic head geometry (E40 is indeed lateral)
    # A fixed sphere gives the most scientifically accurate and reproducible view
    im, cn = mne.viz.plot_topomap(
        data,
        info,
        axes=ax,
        show=False,
        contours=6,  # Show contour lines for better structure
        cmap='RdYlGn',
        vlim=(0, 1),
        sensors=True,
        names=used_channels,  # Show names of used channels only
        outlines='head',
        sphere=(0, 0, 0, 0.095),  # Fixed EGI-standard sphere for consistent view
        image_interp='cubic',  # Smoother interpolation for clearer visualization
    )
    ax.set_title(f'Channel Selection (n={len(used_channels)}/{len(all_channels)} used)\nGreen=Used, Red=Excluded', 
                 fontsize=11, weight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, aspect=30)
    cbar.set_label('Channel Usage', fontsize=9)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Excluded', 'Used'])
    
    # Overall title
    fig.suptitle(title, fontsize=13, weight='bold', y=0.95)
    
    # Add summary text at bottom
    summary_text = (
        f"Total channels in montage: {len(all_channels)} | "
        f"Used: {len(used_channels)} ({len(used_channels)/len(all_channels)*100:.1f}%) | "
        f"Excluded: {len(all_channels) - len(used_channels)}"
    )
    fig.text(0.5, 0.02, summary_text, ha='center', fontsize=9, style='italic')
    
    # Tight layout
    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    print(f"[channel_viz] Saved topomap: {output_path.name}", flush=True)
    
    plt.close()


def resolve_used_channels_from_config(cfg: dict, montage_path: Path) -> list:
    """
    Resolve which channels are actually used based on config.
    
    This function replicates the logic in epoch_utils.spatial_sample() to determine
    which channels will actually be used in the experiment.
    
    Args:
        cfg: Configuration dictionary with channel selection parameters
        montage_path: Path to .sfp montage file
    
    Returns:
        List of channel names that will be used
    """
    # Load montage to get all available channels
    montage = mne.channels.read_custom_montage(str(montage_path))
    all_channels = montage.ch_names
    
    # Get channel lists from config
    channel_lists = cfg.get('channel_lists', {})
    
    # Start with all channels
    used_channels = set(all_channels)
    
    # Step 1: Apply exclusions
    exclude_channel_list = cfg.get('exclude_channel_list')
    if exclude_channel_list:
        if isinstance(exclude_channel_list, str):
            # Named list
            excl_names = channel_lists.get(exclude_channel_list, [])
        elif isinstance(exclude_channel_list, (list, tuple)):
            excl_names = list(exclude_channel_list)
        else:
            excl_names = []
        
        used_channels -= set(excl_names)
    
    # Step 2: Apply inclusions (if specified, this REPLACES the used set)
    include_channels = cfg.get('include_channels')
    if include_channels:
        if isinstance(include_channels, str):
            # Named list
            incl_names = channel_lists.get(include_channels, [])
        elif isinstance(include_channels, (list, tuple)):
            incl_names = list(include_channels)
        else:
            incl_names = []
        
        # Include channels REPLACES the set (doesn't add to it)
        used_channels = set(incl_names) & set(all_channels)  # Only keep valid channels
    
    # Note: cz_step is NOT handled here as it's a dynamic sampling that happens
    # during actual data loading. This function shows the initial channel selection.
    
    return sorted(list(used_channels))


def save_channel_topomap_for_run(cfg: dict, run_dir: Path, montage_path: Path):
    """
    Create and save a channel usage topomap for a training run.
    
    This is a convenience function that combines channel resolution and topomap creation.
    
    Args:
        cfg: Configuration dictionary
        run_dir: Run directory where topomap will be saved
        montage_path: Path to .sfp montage file
    """
    # Resolve used channels
    used_channels = resolve_used_channels_from_config(cfg, montage_path)
    
    # Create title from run directory name
    title = f"Channel Selection: {run_dir.name}"
    
    # Save topomap
    output_path = run_dir / "channel_selection_topomap.png"
    create_channel_usage_topomap(
        montage_path=montage_path,
        used_channels=used_channels,
        output_path=output_path,
        title=title,
        dpi=300
    )
    
    # Also save a text file with channel list
    channel_list_path = run_dir / "channel_selection.txt"
    with open(channel_list_path, 'w') as f:
        f.write(f"Channel Selection for: {run_dir.name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Montage: {montage_path.name}\n")
        f.write(f"Total channels: {len(mne.channels.read_custom_montage(str(montage_path)).ch_names)}\n")
        f.write(f"Used channels: {len(used_channels)}\n\n")
        
        if cfg.get('include_channels'):
            f.write(f"Config setting: include_channels = {cfg['include_channels']}\n")
        if cfg.get('exclude_channel_list'):
            f.write(f"Config setting: exclude_channel_list = {cfg['exclude_channel_list']}\n")
        
        f.write("\nUsed channel names:\n")
        for ch in used_channels:
            f.write(f"  {ch}\n")
    
    print(f"[channel_viz] Saved channel list: {channel_list_path.name}", flush=True)
