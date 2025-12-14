"""
Paper figure: combined model RDM panels (single output; both PI variants included).

Creates publication-ready figures with rank-normalized (0–1) model RDMs:
- PI(1-4)-ANS
- PI(1-3)-ANS
- ANS (log-ratio)
- Pixel area (e-only target)
- RT landing (subject-level mean RT differences)

Outputs (into output_dir/figures):
- model_rdms_panel_all_models.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

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
from scripts.rsa.naming import prefixed_path, prefixed_title


def _rank_normalize_rdm(mat: np.ndarray) -> np.ndarray:
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


def _write_combined(
    *,
    output_dir: Path,
    stimuli_csv: Path,
    rt_summary_csv: Path,
    codes: Sequence[int],
    pi_ans_boundary: float,
) -> None:
    run_root = output_dir
    codes_int = [int(c) for c in codes]
    labels = [code_to_numerosity(c) for c in codes_int]

    pi_14 = build_pi_ans_rdm(codes_int, pi_min=1, pi_max=4, boundary=float(pi_ans_boundary))
    pi_13 = build_pi_ans_rdm(codes_int, pi_min=1, pi_max=3, boundary=float(pi_ans_boundary))
    pi_24 = build_pi_ans_rdm(codes_int, pi_min=2, pi_max=4, boundary=float(pi_ans_boundary))
    ans = build_ans_log_ratio_rdm(codes_int)
    pixel = build_pixel_rdm_e_only(stimuli_csv, codes_int)
    rt = build_rt_landing_rdm(rt_summary_csv, codes_int)

    models = [
        ("PI(1-4)-ANS", pi_14),
        ("PI(1-3)-ANS", pi_13),
        ("PI(2-4)-ANS", pi_24),
        ("ANS (log-ratio)", ans),
        ("Pixel Area", pixel),
        ("RT Landing", rt),
    ]

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes_flat = list(np.ravel(axes))
    ims = []
    for ax, (title, mat) in zip(axes_flat, models):
        norm = _rank_normalize_rdm(mat)
        display = np.array(norm, copy=True)
        display[np.triu_indices_from(display, k=1)] = 0.0

        im = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0)
        ims.append(im)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Numerosity", fontsize=10)
        ax.set_ylabel("Numerosity", fontsize=10)

    # Only keep y-labels on first column in each row
    for i, ax in enumerate(axes_flat):
        if i not in (0, 3):
            ax.set_ylabel("")

    fig.suptitle(prefixed_title(run_root=run_root, title="Candidate Model RDMs"), fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.88, top=0.90, bottom=0.10, wspace=0.35, hspace=0.40)

    cbar_ax = fig.add_axes([0.90, 0.23, 0.015, 0.55])
    cbar = fig.colorbar(ims[0], cax=cbar_ax)
    cbar.set_label("Rank-normalized dissimilarity (0–1)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(
        prefixed_path(run_root=run_root, kind="figures", stem="model_rdms_panel_all_models", ext=".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)


def write_model_rdms_panels_pi_variants(
    *,
    output_dir: Path,
    stimuli_csv: Path,
    rt_summary_csv: Path,
    codes: Sequence[int] = (11, 22, 33, 44, 55, 66),
    pi_ans_boundary: float = 4.0,
) -> None:
    _write_combined(
        output_dir=output_dir,
        stimuli_csv=stimuli_csv,
        rt_summary_csv=rt_summary_csv,
        codes=codes,
        pi_ans_boundary=float(pi_ans_boundary),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper figure: combined model RDM panels (PI(1-4)-ANS, PI(1-3)-ANS, PI(2-4)-ANS, plus ANS/Pixel/RT).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Run directory to write figures/ into.")
    parser.add_argument("--stimuli-csv", type=Path, default=Path("data") / "stimuli" / "stimuli_analysis.csv")
    parser.add_argument("--rt-summary-csv", type=Path, required=True, help="RT subject-level landing summary CSV.")
    parser.add_argument("--codes", nargs="*", type=int, default=[11, 22, 33, 44, 55, 66])
    parser.add_argument("--pi-ans-boundary", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_model_rdms_panels_pi_variants(
        output_dir=args.output_dir,
        stimuli_csv=args.stimuli_csv,
        rt_summary_csv=args.rt_summary_csv,
        codes=args.codes,
        pi_ans_boundary=float(args.pi_ans_boundary),
    )


if __name__ == "__main__":
    main()


