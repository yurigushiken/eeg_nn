"""
Paper figure: combined model RDM panel (1×4).

Creates a single publication-ready figure with rank-normalized (0–1) model RDMs:
- PI-ANS
- ANS (log-ratio)
- Pixel area (e-only target)
- RT landing (subject-level mean RT differences)

Rank-normalization matches Spearman RSA logic and enables a shared color scale.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

# Add project root to path
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.rdm_models import (
    build_ans_log_ratio_rdm,
    build_pi_ans_rdm,
    build_pixel_rdm_e_only,
    build_rt_landing_rdm,
    code_to_numerosity,
)


def _rank_normalize_rdm(mat: np.ndarray) -> np.ndarray:
    """
    Rank-normalize off-diagonal distances to [0,1] based on the lower triangle.
    Diagonal remains 0.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("RDM must be square")
    n = mat.shape[0]
    tri = np.tril_indices(n, k=-1)
    vals = np.asarray(mat[tri], dtype=float)
    if vals.size == 0:
        out = np.zeros_like(mat, dtype=float)
        np.fill_diagonal(out, 0.0)
        return out

    ranks = rankdata(vals, method="average") - 1.0  # 0..(m-1)
    denom = float(max(1, vals.size - 1))
    norm = ranks / denom

    out = np.zeros_like(mat, dtype=float)
    out[tri] = norm
    out = out + out.T
    np.fill_diagonal(out, 0.0)
    return out


def write_model_rdms_panel(
    *,
    output_dir: Path,
    stimuli_csv: Path,
    rt_summary_csv: Path,
    codes: Sequence[int] = (11, 22, 33, 44, 55, 66),
    pi_ans_boundary: float = 4.0,
) -> None:
    codes = [int(c) for c in codes]
    labels = [code_to_numerosity(c) for c in codes]

    pi = build_pi_ans_rdm(codes, boundary=float(pi_ans_boundary))
    ans = build_ans_log_ratio_rdm(codes)
    pixel = build_pixel_rdm_e_only(stimuli_csv, codes)
    rt = build_rt_landing_rdm(rt_summary_csv, codes)

    models = [
        ("PI-ANS", pi),
        ("ANS (log-ratio)", ans),
        ("Pixel Area", pixel),
        ("RT Landing", rt),
    ]

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    ims = []
    for ax, (title, mat) in zip(axes, models):
        norm = _rank_normalize_rdm(mat)
        display = np.array(norm, copy=True)
        # Mask upper triangle with diagonal color (diagonal is 0 after rank-normalization).
        display[np.triu_indices_from(display, k=1)] = 0.0

        im = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0)
        ims.append(im)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Numerosity", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Numerosity", fontsize=10)
        else:
            ax.set_ylabel("")

    # Layout: reserve space on the right for a shared colorbar (avoid overlap/cropping).
    fig.suptitle("Candidate Model RDMs", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.82, bottom=0.15, wspace=0.35)

    # Shared colorbar in a dedicated axis to the right.
    cbar_ax = fig.add_axes([0.90, 0.22, 0.015, 0.56])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label("Rank-normalized dissimilarity (0–1)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    out_path = fig_dir / "model_rdms_panel.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper figure: combined model RDM panel.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory to write figures/ into.")
    parser.add_argument("--stimuli-csv", type=Path, default=Path("data") / "stimuli" / "stimuli_analysis.csv")
    parser.add_argument("--rt-summary-csv", type=Path, required=True, help="RT subject-level landing summary CSV.")
    parser.add_argument("--codes", nargs="*", type=int, default=[11, 22, 33, 44, 55, 66])
    parser.add_argument("--pi-ans-boundary", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_model_rdms_panel(
        output_dir=args.output_dir,
        stimuli_csv=args.stimuli_csv,
        rt_summary_csv=args.rt_summary_csv,
        codes=args.codes,
        pi_ans_boundary=float(args.pi_ans_boundary),
    )


if __name__ == "__main__":
    main()


