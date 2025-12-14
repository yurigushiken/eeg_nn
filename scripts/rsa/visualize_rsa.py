"""
Visualize RSA matrix outputs as a heatmap and 2D MDS scatter plot.

The script reads rsa_matrix_results.csv (as produced by run_rsa_matrix.py),
builds a symmetric accuracy matrix, and generates:
  1) A representational dissimilarity matrix (RDM) heatmap.
  2) A 2D embedding via Multi-Dimensional Scaling (MDS).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

# Ensure project root is importable when running as a script.
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.naming import prefixed_path, prefixed_title

def build_accuracy_matrix(
    csv_path: Path,
    metric: str = "Accuracy",
    subject_filter: Optional[str] = "OVERALL",
) -> Tuple[np.ndarray, List[int]]:
    """
    Build a symmetric matrix for the given metric averaged across seeds.

    Args:
        csv_path: Path to rsa_matrix_results.csv.
        metric: Column name to pivot (e.g., 'Accuracy', 'MacroF1').
        subject_filter: Which subject identifier to keep (e.g., 'OVERALL').
            If None, all rows are used. When the CSV lacks a 'Subject' column,
            all rows are treated as OVERALL-compatible.

    Returns:
        matrix: N×N numpy array with symmetric values.
        labels: Ordered list of condition codes used for rows/columns.
    """
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in {csv_path}. Available: {list(df.columns)}")

    if subject_filter is not None and "Subject" in df.columns:
        df = df[df["Subject"] == subject_filter].copy()
    elif subject_filter is not None and "Subject" not in df.columns and subject_filter != "OVERALL":
        raise ValueError(
            f"Subject filtering requested ('{subject_filter}') but CSV lacks a 'Subject' column."
        )

    grouped = (
        df.groupby(["ClassA", "ClassB"], as_index=False)[metric]
        .mean()
        .sort_values(["ClassA", "ClassB"])
    )

    labels = sorted(set(grouped["ClassA"]).union(grouped["ClassB"]))
    size = len(labels)
    label_to_idx = {lbl: idx for idx, lbl in enumerate(labels)}

    matrix = np.full((size, size), np.nan, dtype=float)
    for _, row in grouped.iterrows():
        a = int(row["ClassA"])
        b = int(row["ClassB"])
        value = float(row[metric])
        i, j = label_to_idx[a], label_to_idx[b]
        matrix[i, j] = value
        matrix[j, i] = value

    if size > 0:
        if metric.lower().startswith("acc"):
            diag_value = 100.0
        else:
            diag_value = float(np.nanmax(matrix))
            if np.isnan(diag_value):
                diag_value = 0.0
        np.fill_diagonal(matrix, diag_value)

    return matrix, labels


def compute_mds_positions(matrix: np.ndarray, labels: Iterable[int]) -> pd.DataFrame:
    """
    Compute a 2D MDS embedding from the accuracy matrix.

    Args:
        matrix: Symmetric accuracy matrix.
        labels: Iterable of condition codes corresponding to matrix rows.

    Returns:
        DataFrame with columns ['label', 'x', 'y'].
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square for MDS.")

    # Convert accuracy to dissimilarity: higher accuracy -> larger distance separation.
    filled = np.array(matrix, copy=True)
    if np.isnan(filled).any():
        mean_val = np.nanmean(filled[~np.isnan(filled)])
        mean_val = float(mean_val if not np.isnan(mean_val) else 0.0)
        filled = np.where(np.isnan(filled), mean_val, filled)

    distances = filled - 50.0
    distances = np.where(distances < 0.0, 0.0, distances)
    np.fill_diagonal(distances, 0.0)

    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        n_init=10,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(distances)
    clean_labels = []
    for lbl in labels:
        try:
            clean_labels.append(int(lbl))
        except (TypeError, ValueError):
            clean_labels.append(lbl)
    return pd.DataFrame({"label": clean_labels, "x": coords[:, 0], "y": coords[:, 1]})


def plot_rdm_heatmap(
    matrix: np.ndarray,
    labels: List[int],
    output_path: Path,
    vmin: float = 50.0,
    vmax: float = 80.0,
    title: str = "RSA Matrix (Higher = Easier to Distinguish)",
) -> None:
    """Plot and save the RDM heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    matrix = np.array(matrix, copy=True)

    mask = np.isnan(matrix)
    off_diag = matrix[~mask]
    fill_value = float(np.nanmean(off_diag)) if off_diag.size else 0.0
    display_matrix = np.where(mask, fill_value, matrix)

    # Mask upper triangle (redundant for symmetric matrices) using diagonal color/value.
    # We do this in display space only, so downstream numeric summaries are unaffected.
    diag_value = float(display_matrix[0, 0]) if display_matrix.size else fill_value
    upper = np.triu(np.ones_like(display_matrix, dtype=bool), k=1)
    display_matrix[upper] = diag_value

    cmap = plt.cm.Blues
    im = ax.imshow(display_matrix, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Class B")
    ax.set_ylabel("Class A")
    ax.set_title(title)

    # Annotate values
    for i in range(len(labels)):
        for j in range(len(labels)):
            if j > i:
                # Upper triangle is intentionally blocked out.
                continue
            value = matrix[i, j]
            if np.isnan(value):
                continue
            text = f"{value:.1f}"
            color = "white" if i == j else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Accuracy / Metric Value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mds_scatter(
    positions: pd.DataFrame,
    output_path: Path,
    flip_xy: bool = False,
    title: str = "MDS Projection of RSA Matrix",
) -> None:
    """Plot and save the 2D MDS scatter plot.

    Args:
        positions: DataFrame with x, y coordinates
        output_path: Path to save PNG
        flip_xy: If True, swap x and y axes for alternative view
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))

    # Optionally flip x and y for alternative visualization
    if flip_xy:
        x_coord = positions["y"]
        y_coord = positions["x"]
    else:
        x_coord = positions["x"]
        y_coord = positions["y"]

    ax.scatter(x_coord, y_coord, s=80, c="#2a6f97")
    for _, row in positions.iterrows():
        label = row["label"]
        if isinstance(label, (float, np.floating)) and float(label).is_integer():
            label_text = str(int(label))
        else:
            label_text = str(label)
        x_pos = row["y"] if flip_xy else row["x"]
        y_pos = row["x"] if flip_xy else row["y"]
        ax.annotate(label_text, (x_pos, y_pos), textcoords="offset points", xytext=(5, 5))

    # Set x-axis range and ticks based on whether flipped
    if flip_xy:
        ax.set_xlim(-15, 15)
        ax.set_xticks(np.arange(-10, 11, 5))  # Labels only from -10 to 10, step 5
        ax.set_ylim(-8, 8)  # Fixed y-axis range for flipped version
    else:
        ax.set_xlim(-7.5, 7.5)
        ax.set_xticks(np.arange(-7.5, 7.6, 2.5))
        # Auto y-axis for non-flipped
        if not positions.empty:
            coord_min, coord_max = y_coord.min(), y_coord.max()
            coord_pad = max((coord_max - coord_min) * 0.02, 0.2)
            ax.set_ylim(coord_min - coord_pad, coord_max + coord_pad)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    if flip_xy:
        ax.set_xlabel("MDS Dimension 2")
        ax.set_ylabel("MDS Dimension 1")
        ax.set_title(title)
    else:
        ax.set_xlabel("MDS Dimension 1")
        ax.set_ylabel("MDS Dimension 2")
        ax.set_title(title)

    ax.set_aspect("equal")

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize RSA matrix results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results") / "runs" / "rsa_results_master.csv",
        help="Path to rsa_results_master.csv.",
    )
    parser.add_argument(
        "--metric",
        default="Accuracy",
        help="Metric column to visualize (default: Accuracy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run root directory to store generated figures (defaults to csv_dir). Figures go into <run_root>/figures.",
    )
    parser.add_argument(
        "--subject",
        default="OVERALL",
        help="Subject identifier to visualize (default: OVERALL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = args.csv
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    matrix, labels = build_accuracy_matrix(csv_path, metric=args.metric, subject_filter=args.subject)
    positions = compute_mds_positions(matrix, labels)

    run_root = args.output_dir if args.output_dir is not None else csv_path.parent
    run_root.mkdir(parents=True, exist_ok=True)
    heatmap_path = prefixed_path(run_root=run_root, kind="figures", stem="brain_rdm_heatmap", ext=".png")
    scatter_path = prefixed_path(run_root=run_root, kind="figures", stem="mds", ext=".png")

    plot_rdm_heatmap(
        matrix,
        labels,
        heatmap_path,
        title=prefixed_title(run_root=run_root, title="RSA Matrix (Higher = Easier to Distinguish)"),
    )
    plot_mds_scatter(
        positions,
        scatter_path,
        flip_xy=True,
        title=prefixed_title(run_root=run_root, title="MDS Projection of RSA Matrix"),
    )

    print(f"[visualize_rsa] Heatmap saved to {heatmap_path}")
    print(f"[visualize_rsa] MDS scatter (flipped) saved to {scatter_path}")


if __name__ == "__main__":
    main()

