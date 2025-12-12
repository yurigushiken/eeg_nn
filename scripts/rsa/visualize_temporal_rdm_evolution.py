"""
Visualize temporal evolution of RDMs (Representational Dissimilarity Matrices).

Creates:
1. Animated GIF showing RDM evolution across all 23 time windows
2. Snapshot figure with 6 key timepoints for publication

RDM Structure:
- 6x6 matrix (numerosities 1-6)
- Cell values = pairwise discrimination accuracy (50-100%)
- Diagonal = 100% (self-discrimination, not used)
- Off-diagonal = classification accuracy for each pair
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

# Matplotlib settings for publication quality
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 9
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['figure.dpi'] = 300


def create_rdm_at_timepoint(
    subject_data: pd.DataFrame,
    time_center: float,
    numerosities: List[int] = [11, 22, 33, 44, 55, 66]
) -> np.ndarray:
    """
    Create RDM (6x6) at a specific time window.

    Args:
        subject_data: Subject-level temporal data (seeds averaged)
        time_center: Time window center (ms)
        numerosities: List of numerosity codes (default: [11,22,33,44,55,66])

    Returns:
        6x6 numpy array with pairwise accuracies
    """
    n_nums = len(numerosities)
    rdm = np.full((n_nums, n_nums), np.nan)

    # Fill diagonal with 100% (self-discrimination)
    np.fill_diagonal(rdm, 100.0)

    # Filter to this timepoint
    time_data = subject_data[subject_data['TimeWindow_Center'] == time_center]

    # Fill off-diagonal with pairwise accuracies
    for i, num_a in enumerate(numerosities):
        for j, num_b in enumerate(numerosities):
            if i >= j:
                continue  # Skip diagonal and lower triangle

            # Get accuracy for this pair
            pair_data = time_data[
                (time_data['ClassA'] == num_a) &
                (time_data['ClassB'] == num_b)
            ]

            if len(pair_data) > 0:
                # Mean across subjects
                mean_acc = pair_data['Accuracy'].mean()
                rdm[i, j] = mean_acc
                rdm[j, i] = mean_acc  # Symmetric

    return rdm


def plot_rdm(
    rdm: np.ndarray,
    ax: plt.Axes,
    time_center: float,
    numerosities: List[int] = [11, 22, 33, 44, 55, 66],
    vmin: float = 50.0,
    vmax: float = 80.0,
    show_colorbar: bool = True,
    title_prefix: str = ""
) -> None:
    """
    Plot a single RDM heatmap.

    Args:
        rdm: 6x6 RDM matrix
        ax: Matplotlib axes
        time_center: Time window center (ms)
        numerosities: List of numerosity codes
        vmin: Colorbar minimum
        vmax: Colorbar maximum
        show_colorbar: Whether to show colorbar
        title_prefix: Optional prefix for title
    """
    # Mask diagonal (self-discrimination)
    rdm_masked = rdm.copy()
    np.fill_diagonal(rdm_masked, np.nan)

    # Heatmap
    im = ax.imshow(
        rdm_masked,
        cmap='Blues',
        vmin=vmin,
        vmax=vmax,
        aspect='equal',
        origin='upper',
        interpolation='nearest'
    )

    # Labels
    ax.set_xticks(np.arange(len(numerosities)))
    ax.set_yticks(np.arange(len(numerosities)))
    ax.set_xticklabels(numerosities, fontsize=9)
    ax.set_yticklabels(numerosities, fontsize=9)

    ax.set_xlabel('Numerosity', fontsize=10)
    ax.set_ylabel('Numerosity', fontsize=10)

    # Title
    title = f"{title_prefix}{int(time_center)}ms"
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Add text annotations (accuracy values)
    for i in range(len(numerosities)):
        for j in range(len(numerosities)):
            if i == j:
                continue  # Skip diagonal
            if not np.isnan(rdm_masked[i, j]):
                text_color = 'white' if rdm_masked[i, j] > 65 else 'black'
                ax.text(
                    j, i, f"{rdm_masked[i, j]:.1f}",
                    ha="center", va="center",
                    color=text_color,
                    fontsize=7
                )

    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Accuracy (%)', rotation=270, labelpad=15, fontsize=9)

    # Grid
    ax.set_xticks(np.arange(len(numerosities) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(numerosities) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)


def create_rdm_snapshot_grid(
    subject_data: pd.DataFrame,
    output_path: Path,
    timepoints: List[float] = [50, 100, 150, 200, 300, 400],
    numerosities: List[int] = [11, 22, 33, 44, 55, 66]
) -> None:
    """
    Create grid of RDM snapshots at key timepoints.

    Args:
        subject_data: Subject-level temporal data
        output_path: Output file path
        timepoints: List of time centers to display (ms)
        numerosities: List of numerosity codes
    """
    # Find closest actual timepoints
    available_times = sorted(subject_data['TimeWindow_Center'].unique())
    selected_times = []
    for target_t in timepoints:
        closest_t = min(available_times, key=lambda t: abs(t - target_t))
        selected_times.append(closest_t)

    n_times = len(selected_times)

    # Create figure
    fig = plt.figure(figsize=(15, 6))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.4)

    for idx, time_center in enumerate(selected_times):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        # Create RDM at this timepoint
        rdm = create_rdm_at_timepoint(
            subject_data=subject_data,
            time_center=time_center,
            numerosities=numerosities
        )

        # Plot
        plot_rdm(
            rdm=rdm,
            ax=ax,
            time_center=time_center,
            numerosities=numerosities,
            vmin=50.0,
            vmax=80.0,
            show_colorbar=(col == 2),  # Only show colorbar on rightmost plots
            title_prefix=""
        )

    fig.suptitle('Temporal RDM Evolution: Key Snapshots', fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[viz] Saved: {output_path}")
    plt.close()


def create_rdm_evolution_gif(
    subject_data: pd.DataFrame,
    output_path: Path,
    numerosities: List[int] = [11, 22, 33, 44, 55, 66],
    fps: int = 4
) -> None:
    """
    Create animated GIF showing RDM evolution across all time windows.

    Args:
        subject_data: Subject-level temporal data
        output_path: Output GIF path
        numerosities: List of numerosity codes
        fps: Frames per second for GIF
    """
    # Get all time windows (sorted)
    time_windows = sorted(subject_data['TimeWindow_Center'].unique())
    n_frames = len(time_windows)

    print(f"[viz] Creating animated GIF with {n_frames} frames...")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    def update_frame(frame_idx):
        """Update function for animation."""
        ax.clear()

        time_center = time_windows[frame_idx]

        # Create RDM at this timepoint
        rdm = create_rdm_at_timepoint(
            subject_data=subject_data,
            time_center=time_center,
            numerosities=numerosities
        )

        # Plot
        plot_rdm(
            rdm=rdm,
            ax=ax,
            time_center=time_center,
            numerosities=numerosities,
            vmin=50.0,
            vmax=80.0,
            show_colorbar=True,
            title_prefix="Time: "
        )

        # Add frame counter (using ax.text so it gets cleared with ax.clear())
        ax.text(
            0.98, -0.15,
            f"Frame {frame_idx + 1}/{n_frames}",
            transform=ax.transAxes,
            ha='right',
            fontsize=8,
            color='gray'
        )

    # Create animation
    anim = FuncAnimation(
        fig,
        update_frame,
        frames=n_frames,
        interval=1000 // fps,  # milliseconds per frame
        repeat=True
    )

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)

    print(f"[viz] Saved: {output_path}")
    plt.close()


def create_rdm_evolution_summary(
    subject_data: pd.DataFrame,
    output_path: Path,
    numerosities: List[int] = [11, 22, 33, 44, 55, 66]
) -> None:
    """
    Create summary figure showing how each pair's accuracy evolves over time.

    Args:
        subject_data: Subject-level temporal data
        output_path: Output file path
        numerosities: List of numerosity codes
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get all time windows
    time_windows = sorted(subject_data['TimeWindow_Center'].unique())

    # Plot each pair's temporal trajectory
    for i, num_a in enumerate(numerosities):
        for j, num_b in enumerate(numerosities):
            if i >= j:
                continue  # Skip diagonal and duplicates

            # Get temporal trajectory for this pair
            pair_data = subject_data[
                (subject_data['ClassA'] == num_a) &
                (subject_data['ClassB'] == num_b)
            ]

            # Group by time window and compute mean
            temporal_curve = pair_data.groupby('TimeWindow_Center')['Accuracy'].mean()

            # Plot
            ratio = num_b / num_a
            alpha = 0.7 if ratio >= 2.0 else 0.4  # Highlight easy pairs
            linewidth = 2.0 if ratio >= 2.0 else 1.0

            ax.plot(
                temporal_curve.index,
                temporal_curve.values,
                label=f'{num_a}v{num_b} (r={ratio:.2f})',
                alpha=alpha,
                linewidth=linewidth
            )

    # Chance level
    ax.axhline(50, color='black', linestyle='--', linewidth=1, label='Chance', zorder=1)

    # Labels and formatting
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Temporal Evolution: All Pairwise Discriminations', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 500)
    ax.set_ylim(45, 85)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[viz] Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize temporal RDM evolution"
    )
    parser.add_argument(
        "--subject-data",
        type=Path,
        required=True,
        help="Subject-level temporal data (seeds averaged)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for figures"
    )
    parser.add_argument(
        "--create-gif",
        action="store_true",
        help="Create animated GIF (slower, ~1-2 minutes)"
    )
    parser.add_argument(
        "--snapshot-times",
        type=float,
        nargs='+',
        default=[50, 100, 150, 200, 300, 400],
        help="Time points for snapshot grid (ms)"
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=4,
        help="Frames per second for GIF (default: 4)"
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("  TEMPORAL RSA: RDM EVOLUTION VISUALIZATION")
    print("="*70)

    # Load data
    print(f"[viz] Loading subject data...")
    subject_df = pd.read_csv(args.subject_data)

    # Create snapshot grid
    print(f"[viz] Creating RDM snapshot grid...")
    output_snapshots = args.output_dir / "temporal_rdm_snapshots.png"
    create_rdm_snapshot_grid(
        subject_data=subject_df,
        output_path=output_snapshots,
        timepoints=args.snapshot_times
    )

    # Create summary trajectory plot
    print(f"[viz] Creating temporal trajectory summary...")
    output_summary = args.output_dir / "temporal_rdm_trajectories.png"
    create_rdm_evolution_summary(
        subject_data=subject_df,
        output_path=output_summary
    )

    # Create animated GIF (optional, slower)
    if args.create_gif:
        print(f"[viz] Creating animated GIF (this may take 1-2 minutes)...")
        output_gif = args.output_dir / "temporal_rdm_evolution.gif"
        create_rdm_evolution_gif(
            subject_data=subject_df,
            output_path=output_gif,
            fps=args.gif_fps
        )

    print("="*70)
    print("  [OK] COMPLETE: RDM evolution visualizations generated")
    print("="*70)


if __name__ == "__main__":
    main()
