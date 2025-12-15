"""
Temporal generalization (train-time × test-time) post-training analysis (core).

Core logic lives here so the CLI wrapper stays small.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.rsa.analyze_temporal_stats import apply_fdr_correction
from scripts.rsa.naming import prefixed_path
from scripts.rsa.temporal_paper_utils import sem, timepoint_edges, ttest_1samp_greater


_RUN_RE = re.compile(
    r".*_rsa_(?P<a>-?\d+)v(?P<b>-?\d+)_seed_(?P<seed>-?\d+)_t(?P<t0>-?\d+)-(?P<t1>-?\d+)ms$"
)


def parse_run_dir_name(run_dir_name: str) -> Tuple[int, int, int, int, int]:
    m = _RUN_RE.match(run_dir_name)
    if not m:
        raise ValueError(f"Unrecognized run dir name format: {run_dir_name!r}")
    return (
        int(m.group("a")),
        int(m.group("b")),
        int(m.group("seed")),
        int(m.group("t0")),
        int(m.group("t1")),
    )


def _code_to_label(code: int) -> str:
    if code != 0 and (code % 11 == 0):
        return str(code // 11)
    return str(code)

def _code_to_int_label(code: int) -> int:
    """Integer label for plotting grid axes (prefer 1..6 when encoded as 11..66)."""
    return int(code // 11) if (code != 0 and (code % 11 == 0)) else int(code)


def pair_label(class_a: int, class_b: int) -> str:
    return f"{_code_to_label(int(class_a))}v{_code_to_label(int(class_b))}"


def _iter_completed_run_dirs(run_root: Path) -> Iterable[Path]:
    for d in sorted(run_root.iterdir()):
        if not d.is_dir():
            continue
        pred = d / "test_predictions_outer_generalization.csv"
        met = d / "outer_eval_metrics_generalization.csv"
        if pred.exists() and met.exists():
            yield d


def summarize_run_predictions(run_dir: Path) -> pd.DataFrame:
    pred_path = run_dir / "test_predictions_outer_generalization.csv"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    usecols = {
        "subject_id",
        "correct",
        "train_window_start",
        "train_window_end",
        "train_window_center",
        "test_window_start",
        "test_window_end",
        "test_window_center",
    }
    df = pd.read_csv(pred_path, usecols=lambda c: c in usecols)
    missing = sorted(list(usecols - set(df.columns)))
    if missing:
        raise KeyError(f"Missing required cols in {pred_path.name}: {missing}")

    tw = df[["train_window_start", "train_window_end", "train_window_center"]].drop_duplicates()
    if len(tw) != 1:
        raise ValueError(f"Expected exactly 1 train window per run, got {len(tw)} in {run_dir.name}")
    train_start = int(tw.iloc[0]["train_window_start"])
    train_end = int(tw.iloc[0]["train_window_end"])
    train_center = float(tw.iloc[0]["train_window_center"])

    out = (
        df.groupby(["subject_id", "test_window_start", "test_window_end", "test_window_center"], as_index=False)
        .agg(
            Accuracy=("correct", lambda x: float(np.mean(x)) * 100.0),
            N_Trials=("correct", "size"),
        )
        .rename(columns={"subject_id": "Subject"})
    )
    out["TrainWindow_Start"] = train_start
    out["TrainWindow_End"] = train_end
    out["TrainWindow_Center"] = train_center
    return out


def build_subject_level_table(run_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run_dir in _iter_completed_run_dirs(run_root):
        a, b, seed, _, _ = parse_run_dir_name(run_dir.name)
        part = summarize_run_predictions(run_dir)
        part["ClassA"] = int(a)
        part["ClassB"] = int(b)
        part["Seed"] = int(seed)
        rows.append(part)

    if not rows:
        raise ValueError(f"No completed run dirs found under {run_root}")
    df = pd.concat(rows, ignore_index=True)

    df["Subject"] = df["Subject"].astype(int)
    df["ClassA"] = df["ClassA"].astype(int)
    df["ClassB"] = df["ClassB"].astype(int)
    df["Seed"] = df["Seed"].astype(int)
    for c in ["TrainWindow_Start", "TrainWindow_End", "test_window_start", "test_window_end"]:
        df[c] = df[c].astype(int)
    for c in ["TrainWindow_Center", "test_window_center", "Accuracy"]:
        df[c] = df[c].astype(float)
    return df


def average_seeds_within_subject(subject_level: pd.DataFrame) -> pd.DataFrame:
    return (
        subject_level.groupby(
            [
                "Subject",
                "ClassA",
                "ClassB",
                "TrainWindow_Start",
                "TrainWindow_End",
                "TrainWindow_Center",
                "test_window_start",
                "test_window_end",
                "test_window_center",
            ],
            as_index=False,
        )
        .agg(
            Accuracy=("Accuracy", "mean"),
            N_Trials=("N_Trials", "sum"),
        )
    )


def average_pairs_within_subject(seed_averaged: pd.DataFrame) -> pd.DataFrame:
    df = seed_averaged.copy()
    df["PairKey"] = df["ClassA"].astype(str) + "v" + df["ClassB"].astype(str)
    return (
        df.groupby(
            [
                "Subject",
                "TrainWindow_Start",
                "TrainWindow_End",
                "TrainWindow_Center",
                "test_window_start",
                "test_window_end",
                "test_window_center",
            ],
            as_index=False,
        )
        .agg(
            Accuracy=("Accuracy", "mean"),
            N_Pairs=("PairKey", "nunique"),
        )
    )


def compute_generalization_matrices(
    subj_cell: pd.DataFrame, *, baseline: float, fdr_alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    train_times = np.sort(subj_cell["TrainWindow_Center"].unique().astype(float))
    test_times = np.sort(subj_cell["test_window_center"].unique().astype(float))
    x_index = {float(t): i for i, t in enumerate(train_times)}
    y_index = {float(t): i for i, t in enumerate(test_times)}

    mean_mat = np.full((len(test_times), len(train_times)), np.nan, dtype=float)
    sem_mat = np.full((len(test_times), len(train_times)), np.nan, dtype=float)

    stats_rows: list[dict] = []
    for (tr_c, te_c), g in subj_cell.groupby(["TrainWindow_Center", "test_window_center"]):
        tr_c_f = float(tr_c)
        te_c_f = float(te_c)
        vals = g["Accuracy"].astype(float).to_numpy()
        t_stat, p_one, n = ttest_1samp_greater(vals, popmean=baseline)
        xi = x_index[tr_c_f]
        yi = y_index[te_c_f]
        mean_mat[yi, xi] = float(np.nanmean(vals)) if vals.size else float("nan")
        sem_mat[yi, xi] = sem(vals)
        stats_rows.append(
            {
                "TrainWindow_Center": tr_c_f,
                "TestWindow_Center": te_c_f,
                "N_Subjects": int(n),
                "Mean_Accuracy": mean_mat[yi, xi],
                "SEM_Accuracy": sem_mat[yi, xi],
                "t_stat": float(t_stat),
                "p_value": float(p_one),
            }
        )

    stats_df = pd.DataFrame(stats_rows).sort_values(["TrainWindow_Center", "TestWindow_Center"]).reset_index(drop=True)

    p_raw = stats_df["p_value"].to_numpy(dtype=float)
    p_for_fdr = np.nan_to_num(p_raw, nan=1.0, posinf=1.0, neginf=1.0)
    p_fdr, rejected = apply_fdr_correction(p_values=p_for_fdr, alpha=fdr_alpha, method="bh")
    stats_df["p_fdr"] = p_fdr
    stats_df["significant_fdr"] = (rejected.astype(bool)) & np.isfinite(p_raw)

    sig_mask = np.full((len(test_times), len(train_times)), False, dtype=bool)
    for _, r in stats_df.iterrows():
        xi = x_index[float(r["TrainWindow_Center"])]
        yi = y_index[float(r["TestWindow_Center"])]
        sig_mask[yi, xi] = bool(r["significant_fdr"])

    return mean_mat, sem_mat, sig_mask, stats_df, train_times, test_times


def _tick_times_for_centers(times: np.ndarray, *, step_ms: float = 100.0, include_last: bool = True) -> np.ndarray:
    """
    Choose major tick locations that align exactly with the actual time-window centers.

    The temporal windows in this project overlap (stride < window), so we plot each
    cell as a sample located at its window center. Ticks placed on those centers
    avoid "between-cell" labels like 100/200 that don't correspond to a plotted bin.
    """
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or t.size == 0:
        return np.asarray([], dtype=float)
    t = np.sort(t)
    t0 = float(t[0])

    step = float(step_ms)
    if step <= 0:
        raise ValueError("step_ms must be > 0")

    chosen: list[float] = []
    for val in t:
        # Keep times that are exactly on the grid defined by (t0 + k*step_ms).
        if np.isclose((val - t0) % step, 0.0, atol=1e-6) or np.isclose((val - t0) % step, step, atol=1e-6):
            chosen.append(float(val))

    if include_last:
        last = float(t[-1])
        if not chosen or not np.isclose(chosen[-1], last, atol=1e-6):
            chosen.append(last)

    return np.asarray(chosen, dtype=float)


def _apply_time_axis_ticks(
    ax: plt.Axes,
    *,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    step_ms: float = 100.0,
) -> None:
    ax.set_xticks(_tick_times_for_centers(x_centers, step_ms=step_ms, include_last=True))
    ax.set_yticks(_tick_times_for_centers(y_centers, step_ms=step_ms, include_last=True))
    ax.set_xticklabels([f"{int(x)}" for x in ax.get_xticks()])
    ax.set_yticklabels([f"{int(y)}" for y in ax.get_yticks()])

    # Minor ticks at cell borders to make the grid align with the binned values.
    ax.set_xticks(x_edges, minor=True)
    ax.set_yticks(y_edges, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.18)
    ax.tick_params(which="minor", length=0)


def _plot_accuracy(*, run_root: Path, mean_mat: np.ndarray, train_times: np.ndarray, test_times: np.ndarray, title: str, stem: str) -> Path:
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    x_edges = timepoint_edges(train_times)
    y_edges = timepoint_edges(test_times)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    m = ax.pcolormesh(x_edges, y_edges, mean_mat, shading="auto", cmap="viridis")
    ax.plot([x_edges[0], x_edges[-1]], [y_edges[0], y_edges[-1]], color="white", linestyle="--", linewidth=1.0, alpha=0.9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training time (ms)")
    ax.set_ylabel("Test time (ms)")
    _apply_time_axis_ticks(ax, x_centers=train_times, y_centers=test_times, x_edges=x_edges, y_edges=y_edges, step_ms=100.0)
    cbar = fig.colorbar(m, ax=ax, shrink=0.95, pad=0.02)
    cbar.set_label("Classifier accuracy (%)")
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    fig.tight_layout()

    out_path = prefixed_path(run_root=run_root, kind="figures", stem=stem, ext=".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_significance(*, run_root: Path, sig_mask: np.ndarray, train_times: np.ndarray, test_times: np.ndarray, title: str, stem: str) -> Path:
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.linewidth"] = 1.0

    x_edges = timepoint_edges(train_times)
    y_edges = timepoint_edges(test_times)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    m = ax.pcolormesh(x_edges, y_edges, sig_mask.astype(float), shading="auto", cmap="Reds", vmin=0.0, vmax=1.0)
    ax.plot([x_edges[0], x_edges[-1]], [y_edges[0], y_edges[-1]], color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Training time (ms)")
    ax.set_ylabel("Test time (ms)")
    _apply_time_axis_ticks(ax, x_centers=train_times, y_centers=test_times, x_edges=x_edges, y_edges=y_edges, step_ms=100.0)
    cbar = fig.colorbar(m, ax=ax, shrink=0.95, pad=0.02, ticks=[0, 1])
    cbar.ax.set_yticklabels(["n.s.", "q<0.05"])
    cbar.set_label("Significance (BH-FDR)")
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_ylim(float(y_edges[0]), float(y_edges[-1]))
    fig.tight_layout()

    out_path = prefixed_path(run_root=run_root, kind="figures", stem=stem, ext=".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_pair_grid_panel(
    *,
    run_root: Path,
    codes: list[int],
    pair_to_png: dict[tuple[int, int], Path],
    title: str,
    stem: str,
) -> Path:
    """
    Create a 6×6 (or N×N) grid where the lower triangle contains the per-pair PNG plot
    and the upper triangle is blank to avoid redundancy.
    """
    labels = sorted({_code_to_int_label(c) for c in codes})
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)

    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.6

    fig, axes = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    if n == 1:
        axes = np.asarray([[axes]])

    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if r < c:
                # Upper triangle: blank (masked redundancy)
                ax.set_facecolor("white")
                ax.axis("off")
                continue

            if r == c:
                # Diagonal: label only
                ax.set_facecolor("white")
                ax.text(0.5, 0.5, str(labels[r]), ha="center", va="center", fontsize=14, fontweight="bold")
                ax.axis("off")
                continue

            # Lower triangle: show the plot image for pair (col_label v row_label)
            a = labels[c]
            b = labels[r]
            key = (a, b) if (a, b) in pair_to_png else (b, a)
            p = pair_to_png.get(key)
            if p is None or not p.exists():
                ax.set_facecolor("#F8F8F8")
                ax.text(0.5, 0.5, f"{a}v{b}\n(missing)", ha="center", va="center", fontsize=8)
                ax.axis("off")
                continue

            img = plt.imread(str(p))
            ax.imshow(img)
            ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    # Tighten margins: these panels can otherwise reserve too much top padding
    # above the grid, especially when saved with bbox_inches="tight".
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.97, wspace=0.02, hspace=0.02)

    out_path = prefixed_path(run_root=run_root, kind="figures", stem=stem, ext=".png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def run_temporal_generalization_analysis(
    *,
    run_root: Path,
    baseline: float = 50.0,
    fdr_alpha: float = 0.05,
    write_overall: bool = True,
    write_per_pair: bool = True,
) -> dict:
    subject_level = build_subject_level_table(run_root)
    out_subject_level = prefixed_path(run_root=run_root, kind="tables", stem="temporal_generalization_subject_level", ext=".csv")
    subject_level.to_csv(out_subject_level, index=False)

    subject_means = average_seeds_within_subject(subject_level)
    out_subject_means = prefixed_path(run_root=run_root, kind="tables", stem="temporal_generalization_subject_means", ext=".csv")
    subject_means.to_csv(out_subject_means, index=False)

    subj_pair_means = average_pairs_within_subject(subject_means)
    out_subj_pair = prefixed_path(run_root=run_root, kind="tables", stem="temporal_generalization_subject_pair_means", ext=".csv")
    subj_pair_means.to_csv(out_subj_pair, index=False)

    figs_per_pair: dict[str, dict[str, str]] = {}
    fig_acc = ""
    fig_sig = ""
    out_stats = ""
    mean_csv = ""
    sem_csv = ""
    mask_csv = ""

    if write_overall:
        mean_mat, sem_mat, sig_mask, stats_df, train_times, test_times = compute_generalization_matrices(
            subj_pair_means, baseline=baseline, fdr_alpha=fdr_alpha
        )
        out_stats = str(prefixed_path(run_root=run_root, kind="tables", stem="temporal_generalization_cell_stats", ext=".csv"))
        stats_df.to_csv(out_stats, index=False)

        mean_csv = str(prefixed_path(run_root=run_root, kind="tables", stem="temporal_generalization_mean_matrix", ext=".csv"))
        sem_csv = str(prefixed_path(run_root=run_root, kind="tables", stem="temporal_generalization_sem_matrix", ext=".csv"))
        mask_csv = str(prefixed_path(run_root=run_root, kind="tables", stem="temporal_generalization_fdr_mask", ext=".csv"))
        pd.DataFrame(mean_mat, index=test_times, columns=train_times).to_csv(mean_csv, index=True)
        pd.DataFrame(sem_mat, index=test_times, columns=train_times).to_csv(sem_csv, index=True)
        pd.DataFrame(sig_mask.astype(int), index=test_times, columns=train_times).to_csv(mask_csv, index=True)

        fig_acc = str(
            _plot_accuracy(
                run_root=run_root,
                mean_mat=mean_mat,
                train_times=train_times,
                test_times=test_times,
                title="Temporal Generalization (Accuracy) — Overall",
                stem="temporal_generalization_accuracy",
            )
        )
        fig_sig = str(
            _plot_significance(
                run_root=run_root,
                sig_mask=sig_mask,
                train_times=train_times,
                test_times=test_times,
                title="Temporal Generalization (Significance) — Overall",
                stem="temporal_generalization_significance",
            )
        )

    if write_per_pair:
        for a, b in (
            subject_means[["ClassA", "ClassB"]].drop_duplicates().sort_values(["ClassA", "ClassB"]).itertuples(index=False, name=None)
        ):
            pl = pair_label(int(a), int(b))
            pair_df = subject_means[(subject_means["ClassA"] == int(a)) & (subject_means["ClassB"] == int(b))].copy()
            # Reduce to subject×cell (defensive: avoid unexpected duplicates).
            pair_df = pair_df.groupby(["Subject", "TrainWindow_Center", "test_window_center"], as_index=False).agg(Accuracy=("Accuracy", "mean"))

            mean_mat_p, sem_mat_p, sig_mask_p, _stats_df_p, train_times_p, test_times_p = compute_generalization_matrices(
                pair_df, baseline=baseline, fdr_alpha=fdr_alpha
            )
            fig_acc_p = str(
                _plot_accuracy(
                    run_root=run_root,
                    mean_mat=mean_mat_p,
                    train_times=train_times_p,
                    test_times=test_times_p,
                    title=f"Temporal Generalization (Accuracy) — {pl}",
                    stem=f"temporal_generalization_accuracy_{pl}",
                )
            )
            fig_sig_p = str(
                _plot_significance(
                    run_root=run_root,
                    sig_mask=sig_mask_p,
                    train_times=train_times_p,
                    test_times=test_times_p,
                    title=f"Temporal Generalization (Significance) — {pl}",
                    stem=f"temporal_generalization_significance_{pl}",
                )
            )
            figs_per_pair[pl] = {"accuracy": fig_acc_p, "significance": fig_sig_p}

        # Pair-grid panels (one for accuracy, one for significance).
        pair_to_acc_png: dict[tuple[int, int], Path] = {}
        pair_to_sig_png: dict[tuple[int, int], Path] = {}
        for a, b in subject_means[["ClassA", "ClassB"]].drop_duplicates().itertuples(index=False, name=None):
            pl = pair_label(int(a), int(b))
            acc = figs_per_pair.get(pl, {}).get("accuracy", "")
            sig = figs_per_pair.get(pl, {}).get("significance", "")
            if acc:
                pair_to_acc_png[(_code_to_int_label(int(a)), _code_to_int_label(int(b)))] = Path(acc)
            if sig:
                pair_to_sig_png[(_code_to_int_label(int(a)), _code_to_int_label(int(b)))] = Path(sig)

        _plot_pair_grid_panel(
            run_root=run_root,
            codes=list(subject_means["ClassA"].unique()) + list(subject_means["ClassB"].unique()),
            pair_to_png=pair_to_acc_png,
            title="Temporal Generalization Pair Grid (Accuracy)",
            stem="temporal_generalization_accuracy_pair_grid",
        )
        _plot_pair_grid_panel(
            run_root=run_root,
            codes=list(subject_means["ClassA"].unique()) + list(subject_means["ClassB"].unique()),
            pair_to_png=pair_to_sig_png,
            title="Temporal Generalization Pair Grid (Significance)",
            stem="temporal_generalization_significance_pair_grid",
        )

    return {
        "subject_level_csv": str(out_subject_level),
        "subject_means_csv": str(out_subject_means),
        "subject_pair_means_csv": str(out_subj_pair),
        "cell_stats_csv": out_stats,
        "mean_matrix_csv": mean_csv,
        "sem_matrix_csv": sem_csv,
        "mask_matrix_csv": mask_csv,
        "fig_accuracy": fig_acc,
        "fig_significance": fig_sig,
        "figures_per_pair": figs_per_pair,
        "n_rows_subject_level": int(len(subject_level)),
    }
