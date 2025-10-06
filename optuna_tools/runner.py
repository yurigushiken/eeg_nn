# optuna_tools/runner.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .config import Config
from .discovery import iter_study_dirs, collect_trials
from .meta import needs_refresh, save_meta
from .cleaning import clean_prefix
from .csv_io import write_trials_csv
from .db import find_optuna_db, generate_plots_from_db
from .reports import write_top3_report, _resolve_confusion_png
from .index_builder import rebuild_index as _rebuild_index


def _pick_best_dirname(df) -> Optional[str]:
    if df is None or df.empty:
        return None
    for col in ("trial_dir", "run_id"):
        if col in df.columns:
            v = df.iloc[0][col]
            if isinstance(v, str) and v:
                return v
    return None


def _sort_trials(df):
    """
    Sort trials by the optimization objective.
    
    Priority:
    1. composite_min_f1_plur_corr (if present - composite objective)
    2. inner_mean_min_per_class_f1 (legacy default)
    3. inner_mean_macro_f1, inner_mean_plur_corr, inner_mean_acc (fallbacks)
    4. mean_acc (last resort)
    
    Scientific rationale: Ranking must align with the actual Optuna objective
    to maintain scientific integrity and proper trial selection.
    """
    # Check for composite objective first (new recommended approach)
    if "composite_min_f1_plur_corr" in df.columns:
        # Filter out None values for composite (only present when that objective is used)
        composite_data = df["composite_min_f1_plur_corr"]
        if composite_data.notna().any():
            return df.sort_values("composite_min_f1_plur_corr", ascending=False).reset_index(drop=True)
    
    # Fallback to legacy ranking order
    preferred_order_cols = [
        "inner_mean_min_per_class_f1",
        "inner_mean_macro_f1", 
        "inner_mean_plur_corr",
        "inner_mean_acc", 
        "mean_acc"
    ]
    for col in preferred_order_cols:
        if col in df.columns:
            return df.sort_values(col, ascending=False).reset_index(drop=True)
    return df


def _flatten_hyper_into_df(df):
    import pandas as pd
    if "hyper" in df.columns:
        hyper_df = pd.json_normalize(df["hyper"])
        df = pd.concat([df.drop(columns=["hyper"]), hyper_df], axis=1)
    return df


def _ensure_run_id(df):
    if "trial_dir" in df.columns:
        cols = ["run_id"] + [c for c in df.columns if c != "run_id"]
        df = df.assign(run_id=df["trial_dir"])[cols]
    return df


def _fallback_history_plot(study_dir: Path, study_name: str, df) -> None:
    import plotly.io as pio
    try:
        import plotly.express as px
    except Exception:
        return
    plots_dir = study_dir / f"!plots-{study_name}"
    plots_dir.mkdir(exist_ok=True)
    x = df["trial_id"] if "trial_id" in df.columns else df.index
    fig = px.scatter(df, x=x, y="mean_acc", title=f"Mean accuracy per trial · {study_name}")
    pio.write_html(fig, plots_dir / f"history_basic-{study_name}.html", include_plotlyjs=True)
    try:
        fig.write_image(plots_dir / f"history_basic-{study_name}.png", scale=2)
    except Exception:
        pass


def _collect_overall_confusions(study_dir: Path, df) -> None:
    """Copy each trial's overall confusion PNG into a per-study folder with unique names.

    Destination: <study_dir>/"!overall confusion"/overall_confusion-<trial_dir_name>.png
    """
    import shutil

    out_dir = study_dir / "!overall confusion"
    out_dir.mkdir(exist_ok=True)

    n_copied = 0
    for _, row in df.iterrows():
        trial_dir_name = str(row.get("trial_dir") or row.get("run_id") or "")
        if not trial_dir_name:
            continue
        trial_dir = study_dir / trial_dir_name
        src = _resolve_confusion_png(trial_dir)
        if not src or not src.exists():
            continue

        stem = f"overall_confusion-{trial_dir_name}"
        dest = out_dir / f"{stem}.png"
        if dest.exists():
            # Ensure uniqueness if a file with that name already exists
            i = 1
            while True:
                alt = out_dir / f"{stem}-{i}.png"
                if not alt.exists():
                    dest = alt
                    break
                i += 1

        try:
            shutil.copyfile(src, dest)
            n_copied += 1
        except Exception as e:
            print(f"  - failed copying confusion for {trial_dir_name}: {e}")

    print(f"  - copied {n_copied} overall confusion PNGs to {out_dir.name}")


def run_refresh(cfg: Config, rebuild_index: bool = False) -> None:
    import shutil
    import pandas as pd

    root = cfg.optuna_root
    root.mkdir(parents=True, exist_ok=True)

    print(f"[start] Refreshing studies in {root} (found {len(list(iter_study_dirs(root)))} pre-scan)")

    for study_dir in iter_study_dirs(root):
        study_name = study_dir.name
        print(f"[scan] {study_name} …")

        records, latest_mtime = collect_trials(study_dir)
        if not records:
            print(f"[empty] {study_name} (no summary_*.json found)")
            continue

        n_trials = len(records)
        if not needs_refresh(study_dir, n_trials, latest_mtime):
            print(f"[skip] {study_name} (up-to-date, n={n_trials})")
            continue

        print(f"[refresh] {study_name} (n={n_trials})")
        clean_prefix(study_dir)

        df = pd.DataFrame(records)
        df = _flatten_hyper_into_df(df)
        df = _sort_trials(df)
        df = _ensure_run_id(df)

        csv_path = write_trials_csv(study_dir, df)
        print(f"  - wrote CSV: {csv_path.name} (rows={len(df)})")

        best_dir_name = _pick_best_dirname(df)
        if best_dir_name:
            best_dir = study_dir / best_dir_name
            html_src = best_dir / "consolidated_report.html"
            pdf_src = best_dir / "consolidated_report.pdf"
            html_dst = study_dir / f"best_consolidated_report-{best_dir_name}.html"
            pdf_dst = study_dir / f"best_consolidated_report-{best_dir_name}.pdf"
            if html_src.exists():
                try:
                    shutil.copyfile(html_src, html_dst)
                    print(f"  - copied best HTML report: {html_dst.name}")
                except Exception as e:
                    print(f"  - failed copying best HTML report: {e}")
            else:
                print(f"  - best HTML report not found in {best_dir_name}")
            if pdf_src.exists():
                try:
                    shutil.copyfile(pdf_src, pdf_dst)
                    print(f"  - copied best PDF report: {pdf_dst.name}")
                except Exception as e:
                    print(f"  - failed copying best PDF report: {e}")
            else:
                print(f"  - best PDF report not found in {best_dir_name}")

        plots_dir = study_dir / f"!plots-{study_name}"

        # Prefer study name from JSON if present
        study_from_json = None
        try:
            study_from_json = records[0].get("study")
        except Exception:
            study_from_json = None

        db = find_optuna_db(cfg.optuna_db_dir, study_from_json or study_name)
        ok = False
        if db is not None:
            print(f"  - using Optuna DB: {db.name}")
            ok = generate_plots_from_db(
                target_study_name=(study_from_json or study_name),
                db_path=db,
                out_dir=plots_dir,
                light_mode=cfg.light_mode,
                n_trials_large=cfg.large_trials_threshold,
                dims_parallel_large=cfg.parallel_dims_large,
                dims_slice_large=cfg.slice_dims_large,
                dims_contour_large=cfg.contour_dims_large,
                skip_png_on_large=cfg.skip_png_on_large,
                force_png_parallel=cfg.force_png_parallel,
                force_png_contour=cfg.force_png_contour,
                png_scale=cfg.png_scale,
            )

        if not ok:
            _fallback_history_plot(study_dir, study_name, df)

        try:
            write_top3_report(study_dir, df, study_name)
        except Exception as e:
            print(f"  - top3 report failed: {e}")

        # Aggregate per-trial overall confusion matrices into a single folder for convenience
        try:
            _collect_overall_confusions(study_dir, df)
        except Exception as e:
            print(f"  - overall confusion aggregation failed: {e}")

        save_meta(study_dir, n_trials, latest_mtime)
        print(f"  - updated meta, latest_mtime={latest_mtime}")

    print("[done] Refresh complete")

    if rebuild_index:
        print("Rebuilding global optuna_runs_index.csv")
        _rebuild_index(cfg)
