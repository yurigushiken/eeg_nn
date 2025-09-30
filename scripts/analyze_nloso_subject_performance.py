#!/usr/bin/env python
from __future__ import annotations

"""
n-LOSO per-subject / per-fold performance analysis.

Creates a new "<run_dir>/subject_performance" folder with:
- per_subject_metrics.csv
- per_fold_metrics.csv
- acc_by_subject_bar.png
- acc_by_fold_bar.png
- overall_confusion.png
- per_subject_confusion/subject-<id>.png
- report.html
- inner_vs_outer/ (if inner predictions CSV is available):
    - inner_vs_outer_subject_metrics.csv
    - inner_vs_outer_scatter.png
- (NEW when inner is available)
    - per_inner_fold_metrics.csv
    - acc_by_inner_fold_bar.png

The script auto-detects these filenames in the run dir:
  Outer (preferred -> fallback): test_predictions_outer.csv -> test_predictions.csv
  Inner (preferred -> fallbacks): test_predictions_inner.csv -> inner_predictions.csv -> val_predictions.csv -> cv_predictions.csv

You can also pass explicit paths with --outer-csv / --inner-csv.
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# -----------------------
# Utilities
# -----------------------

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _auto_find_csv(run_dir: Path, candidates: Iterable[str]) -> Optional[Path]:
    for name in candidates:
        fp = run_dir / name
        if fp.exists():
            return fp
    return None


def _detect_cols(df: pd.DataFrame) -> Tuple[str, str, str, str]:
    """
    Return (subject_col, true_col, pred_col, outer_col).

    - true/pred: prefer *_label_idx then *_label_name if idx not present.
    - outer fold: 'outer_fold' if exists else raise.
    """
    # subject id
    subject_col = None
    for cand in ["subject_id", "subject", "subj", "subjectIDX", "subject_index"]:
        if cand in df.columns:
            subject_col = cand
            break
    if subject_col is None:
        raise ValueError("Could not find a subject column (tried subject_id, subject, subj, subjectIDX, subject_index).")

    # truth
    true_col = None
    for cand in ["true_label_idx", "y_true_idx", "y_true", "true_label_name", "y_true_name"]:
        if cand in df.columns:
            true_col = cand
            break
    if true_col is None:
        raise ValueError("No true label column found (looked for true_label_idx/true_label_name/etc).")

    # pred
    pred_col = None
    for cand in ["pred_label_idx", "y_pred_idx", "y_pred", "pred_label_name", "y_pred_name"]:
        if cand in df.columns:
            pred_col = cand
            break
    if pred_col is None:
        raise ValueError("No predicted label column found (looked for pred_label_idx/pred_label_name/etc).")

    # outer fold
    if "outer_fold" in df.columns:
        outer_col = "outer_fold"
    else:
        raise ValueError("No 'outer_fold' column found in predictions CSV.")

    return subject_col, true_col, pred_col, outer_col


def _coerce_int_if_possible(series: pd.Series) -> pd.Series:
    try:
        return series.astype(int)
    except Exception:
        return series


def _acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean() * 100.0)


# -----------------------
# Core loading
# -----------------------

def _load_predictions(csv_fp: Path) -> Tuple[pd.DataFrame, str, str, str, str, Optional[str]]:
    """
    Load predictions CSV, detect columns, normalize types, and return:
    (df, subject_col, true_col, pred_col, outer_col, inner_col_if_any)
    """
    df = pd.read_csv(csv_fp)
    subject_col, true_col, pred_col, outer_col = _detect_cols(df)
    inner_col = "inner_fold" if "inner_fold" in df.columns else None

    # Normalize types
    df[outer_col] = _coerce_int_if_possible(df[outer_col])
    if inner_col is not None:
        df[inner_col] = _coerce_int_if_possible(df[inner_col])

    # Prefer integer encodings if available for labels
    if true_col.endswith("_idx") or true_col in {"y_true_idx"}:
        df[true_col] = _coerce_int_if_possible(df[true_col])
    if pred_col.endswith("_idx") or pred_col in {"y_pred_idx"}:
        df[pred_col] = _coerce_int_if_possible(df[pred_col])

    # correctness flag if missing
    if "correct" not in df.columns:
        try:
            df["correct"] = (df[pred_col] == df[true_col]).astype(int)
        except Exception:
            df["correct"] = np.nan

    return df, subject_col, true_col, pred_col, outer_col, inner_col


# -----------------------
# Metrics + Plots
# -----------------------

def _per_subject_metrics(df: pd.DataFrame, subject_col: str, true_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for sid, grp in df.groupby(subject_col):
        true = grp[true_col].to_numpy()
        pred = grp[pred_col].to_numpy()
        rows.append({
            "subject_id": sid,
            "n": int(len(grp)),
            "acc_pct": _acc(true, pred)
        })
    out = pd.DataFrame(rows).sort_values(["acc_pct", "n", "subject_id"], ascending=[False, False, True])
    return out


def _per_fold_metrics(df: pd.DataFrame, fold_col: str, true_col: str, pred_col: str) -> pd.DataFrame:
    rows = []
    for f, grp in df.groupby(fold_col):
        rows.append({
            "fold": int(f),
            "n": int(len(grp)),
            "acc_pct": _acc(grp[true_col].to_numpy(), grp[pred_col].to_numpy())
        })
    out = pd.DataFrame(rows).sort_values(["acc_pct", "fold"], ascending=[False, True])
    return out


def _plot_bar(values_df: pd.DataFrame, x_col: str, y_col: str, n_col: str, title: str, xlabel: str, out_png: Path):
    fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=150)
    ax.bar(values_df[x_col].astype(str), values_df[y_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    for idx, row in values_df.reset_index(drop=True).iterrows():
        ax.text(idx, row[y_col] + 1.0, f"{int(row[n_col])}", ha="center", va="bottom", fontsize=8, rotation=0)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _plot_overall_confusion(df: pd.DataFrame, true_col: str, pred_col: str, out_png: Path):
    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy()

    if np.issubdtype(y_true.dtype, np.integer) and np.issubdtype(y_pred.dtype, np.integer):
        labels = sorted({*y_true.tolist(), *y_pred.tolist()})
        xticklabels = [str(x) for x in labels]
        yticklabels = [str(x) for x in labels]
    else:
        labels = sorted({*map(str, y_true.tolist()), *map(str, y_pred.tolist())})
        xticklabels = labels
        yticklabels = labels

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    denom = cm.sum(axis=1, keepdims=True)
    denom[denom == 0] = 1
    cm_pct = cm.astype(float) / denom * 100.0

    fig, ax = plt.subplots(figsize=(6.5, 5.9), dpi=150)
    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        ylabel="True",
        xlabel="Predicted",
        title="Overall Confusion (row-normalized %)"
    )
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}", ha="center", va="center", fontsize=8, color="black")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _per_subject_confusions(df: pd.DataFrame, subject_col: str, true_col: str, pred_col: str, out_dir: Path):
    _ensure_dir(out_dir)
    for sid, grp in df.groupby(subject_col):
        if len(grp) == 0:
            continue
        out_png = out_dir / f"subject-{sid}.png"
        _plot_overall_confusion(grp, true_col, pred_col, out_png)


def _inner_vs_outer_compare(
    df_outer: pd.DataFrame,
    df_inner: pd.DataFrame,
    subject_col: str,
    true_col: str,
    pred_col: str,
    out_dir: Path
):
    _ensure_dir(out_dir)

    per_subj_outer = _per_subject_metrics(df_outer, subject_col, true_col, pred_col)
    per_subj_outer = per_subj_outer.rename(columns={"acc_pct": "outer_acc_pct", "n": "outer_n"})

    per_subj_inner = _per_subject_metrics(df_inner, subject_col, true_col, pred_col)
    per_subj_inner = per_subj_inner.rename(columns={"acc_pct": "inner_acc_pct", "n": "inner_n"})

    joined = per_subj_outer.merge(per_subj_inner, on="subject_id", how="inner")
    joined["delta_outer_minus_inner"] = joined["outer_acc_pct"] - joined["inner_acc_pct"]
    joined["abs_delta"] = joined["delta_outer_minus_inner"].abs()

    csv_fp = out_dir / "inner_vs_outer_subject_metrics.csv"
    joined.sort_values("outer_acc_pct", ascending=False).to_csv(csv_fp, index=False)

    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=150)
    ax.scatter(joined["inner_acc_pct"], joined["outer_acc_pct"])
    for _, r in joined.iterrows():
        ax.text(r["inner_acc_pct"] + 0.2, r["outer_acc_pct"] + 0.2, str(r["subject_id"]), fontsize=7)
    ax.plot([0, 100], [0, 100], ls="--", lw=1)
    ax.set_xlabel("Inner accuracy (%)")
    ax.set_ylabel("Outer accuracy (%)")
    ax.set_title("Subject-wise Inner vs Outer accuracy")
    ax.set_xlim(-1, 101)
    ax.set_ylim(-1, 101)
    fig.tight_layout()
    fig.savefig(out_dir / "inner_vs_outer_scatter.png")
    plt.close(fig)


# -----------------------
# Report
# -----------------------

def _write_report(out_dir: Path, have_inner: bool):
    inner_links = (
        '    <p><strong>Inner vs Outer:</strong> <a href="inner_vs_outer/">inner_vs_outer/</a></p>\n'
        '    <p><strong>Per-inner-fold CSV:</strong> <a href="per_inner_fold_metrics.csv">per_inner_fold_metrics.csv</a></p>\n'
    ) if have_inner else ''

    inner_section = (
        '\n  <div class="row">\n'
        '    <h2>Accuracy by Inner Fold</h2>\n'
        '    <img src="acc_by_inner_fold_bar.png" alt="Accuracy by inner fold">\n'
        '  </div>\n'
    ) if have_inner else ''

    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en"><head>\n'
        '  <meta charset="utf-8">\n'
        '  <title>n-LOSO Subject/Fold Performance</title>\n'
        "  <style>\n"
        "    body { background:#101316; color:#eee; font-family:system-ui, Arial, sans-serif; }\n"
        "    a { color:#9ad; }\n"
        "    .container { padding:16px 18px 24px; }\n"
        "    h1 { margin-top:0; }\n"
        "    img { width:100%; height:auto; border:1px solid #444; }\n"
        "    .row { margin:22px 0; }\n"
        "    .links a { margin-right:16px; }\n"
        "    hr { border:0; border-top:1px solid #333; margin:20px 0; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        '<div class="container">\n'
        "  <h1>n-LOSO Subject/Fold Performance</h1>\n\n"
        '  <div class="links">\n'
        '    <p><strong>Per-subject CSV:</strong> <a href="per_subject_metrics.csv">per_subject_metrics.csv</a></p>\n'
        '    <p><strong>Per-outer-fold CSV:</strong> <a href="per_fold_metrics.csv">per_fold_metrics.csv</a></p>\n'
        '    <p><strong>Per-subject confusion:</strong> <a href="per_subject_confusion/">per_subject_confusion/</a></p>\n'
        f"{inner_links}"
        "  </div>\n\n"
        '  <div class="row">\n'
        "    <h2>Accuracy by Subject (outer)</h2>\n"
        '    <img src="acc_by_subject_bar.png" alt="Accuracy by subject (outer)">\n'
        "  </div>\n\n"
        '  <div class="row">\n'
        "    <h2>Accuracy by Outer Fold</h2>\n"
        '    <img src="acc_by_fold_bar.png" alt="Accuracy by outer fold">\n'
        "  </div>\n"
        f"{inner_section}"
        '  <div class="row">\n'
        "    <h2>Overall Confusion (outer)</h2>\n"
        '    <img src="overall_confusion.png" alt="Overall confusion">\n'
        "  </div>\n\n"
        "</div>\n"
        "</body>\n"
        "</html>\n"
    )
    (out_dir / "report.html").write_text(html, encoding="utf-8")


# -----------------------
# Main analysis
# -----------------------

def analyze_run(run_dir: Path,
                outer_csv: Optional[Path] = None,
                inner_csv: Optional[Path] = None) -> Path:
    out_dir = run_dir / "subject_performance"
    _ensure_dir(out_dir)

    # Auto-discover new names (preferred) then fallbacks
    if outer_csv is None:
        outer_csv = _auto_find_csv(run_dir, [
            "test_predictions_outer.csv",
            "test_predictions.csv",
        ])
    if outer_csv is None:
        raise FileNotFoundError("Could not find an outer predictions CSV (looked for test_predictions_outer.csv, test_predictions.csv).")

    if inner_csv is None:
        inner_csv = _auto_find_csv(run_dir, [
            "test_predictions_inner.csv",
            "inner_predictions.csv",
            "val_predictions.csv",
            "cv_predictions.csv",
        ])

    # Load outer
    df_outer, subject_col, true_col, pred_col, outer_col, _ = _load_predictions(outer_csv)

    # Per-subject metrics (outer)
    per_subj = _per_subject_metrics(df_outer, subject_col, true_col, pred_col)
    per_subj.to_csv(out_dir / "per_subject_metrics.csv", index=False)

    # Per-outer-fold metrics
    per_outer_fold = _per_fold_metrics(df_outer, outer_col, true_col, pred_col)
    per_outer_fold.to_csv(out_dir / "per_fold_metrics.csv", index=False)

    # Plots (outer)
    sb = per_subj.copy()
    sb["subject_id_str"] = sb["subject_id"].astype(str)
    _plot_bar(
        sb.reset_index(drop=True),
        x_col="subject_id_str",
        y_col="acc_pct",
        n_col="n",
        title="n-LOSO: Accuracy by subject (bars labeled with support n)",
        xlabel="Subject",
        out_png=out_dir / "acc_by_subject_bar.png"
    )

    _plot_bar(
        per_outer_fold.reset_index(drop=True),
        x_col="fold",
        y_col="acc_pct",
        n_col="n",
        title="n-LOSO: Accuracy by outer fold (bars labeled with support n)",
        xlabel="Outer fold",
        out_png=out_dir / "acc_by_fold_bar.png"
    )

    # Overall confusion (outer)
    _plot_overall_confusion(df_outer, true_col, pred_col, out_dir / "overall_confusion.png")

    # Per-subject confusion grids (outer)
    _per_subject_confusions(df_outer, subject_col, true_col, pred_col, out_dir / "per_subject_confusion")

    # Inner metrics/plots (if inner available)
    have_inner = False
    if inner_csv is not None and Path(inner_csv).exists():
        try:
            df_inner, subject_col_i, true_col_i, pred_col_i, _, inner_col_i = _load_predictions(Path(inner_csv))
            if inner_col_i is None:
                raise ValueError("Inner predictions CSV is missing 'inner_fold' column.")
            # Inner vs outer per-subject comparison and scatter
            have_inner = True
            _inner_vs_outer_compare(
                df_outer=df_outer,
                df_inner=df_inner,
                subject_col=subject_col,  # assert same column name earlier if you prefer
                true_col=true_col,
                pred_col=pred_col,
                out_dir=out_dir / "inner_vs_outer",
            )
            # Per-inner-fold metrics and plot
            per_inner_fold = _per_fold_metrics(df_inner, inner_col_i, true_col_i, pred_col_i)
            per_inner_fold.to_csv(out_dir / "per_inner_fold_metrics.csv", index=False)
            _plot_bar(
                per_inner_fold.reset_index(drop=True),
                x_col="fold",
                y_col="acc_pct",
                n_col="n",
                title="n-LOSO: Accuracy by inner fold (bars labeled with support n)",
                xlabel="Inner fold",
                out_png=out_dir / "acc_by_inner_fold_bar.png"
            )
        except Exception as e:
            _ensure_dir(out_dir / "inner_vs_outer")
            (out_dir / "inner_vs_outer" / "INFO.txt").write_text(
                f"Could not compute inner-related outputs: {e}",
                encoding="utf-8"
            )

    # HTML report
    _write_report(out_dir, have_inner=have_inner)

    print(f"[ok] Wrote subject/fold performance report to: {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="Analyze n-LOSO per-subject / per-fold performance.")
    ap.add_argument("run_dir", type=str, help="Path to a single run directory (the one that contains test_predictions_* CSVs).")
    ap.add_argument("--outer-csv", type=str, default=None, help="Optional explicit path to the OUTER predictions CSV.")
    ap.add_argument("--inner-csv", type=str, default=None, help="Optional explicit path to the INNER predictions CSV.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outer_csv = Path(args.outer_csv) if args.outer_csv else None
    inner_csv = Path(args.inner_csv) if args.inner_csv else None

    analyze_run(run_dir, outer_csv, inner_csv)


if __name__ == "__main__":
    main()
