from __future__ import annotations
from pathlib import Path
import plotly.io as pio
import optuna
import pandas as pd

from .config import Config
from .discovery import plots_dir as _plots_dir
from .db import load_study_from

def generate_plots_from_db(cfg: Config, study_name: str, db_path: Path, out_dir: Path) -> bool:
    study = load_study_from(cfg, db_path, study_name)
    if study is None:
        return False

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return False

    out_dir.mkdir(exist_ok=True)

    # Param stats
    n_complete = len(trials)
    large = cfg.light_mode and (n_complete >= cfg.large_trials_threshold)
    param_counts: dict[str, int] = {}
    numeric_params: set[str] = set()
    all_params_values: dict[str, list] = {}
    for t in trials:
        for k, v in t.params.items():
            param_counts[k] = param_counts.get(k, 0) + 1
            if isinstance(v, (int, float)):
                numeric_params.add(k)
            all_params_values.setdefault(k, []).append(v)

    sorted_params = [k for k, _ in sorted(param_counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    # unique counts
    try:
        unique_counts: dict[str, int] = {k: len(set(vs)) for k, vs in all_params_values.items()}
    except Exception:
        unique_counts = {k: 2 for k in sorted_params}

    variable_params_sorted = [k for k in sorted_params if unique_counts.get(k, 0) > 1]
    variable_numeric_params_sorted = [k for k in variable_params_sorted if k in numeric_params]
    if not variable_params_sorted:
        variable_params_sorted = list(sorted_params)

    # Caps in large mode; otherwise include all variable params
    params_parallel = (variable_params_sorted[:cfg.parallel_dims_large] if large else variable_params_sorted)
    params_slice = (variable_params_sorted[:cfg.slice_dims_large] if large else variable_params_sorted)
    params_contour = (variable_numeric_params_sorted[:cfg.contour_dims_large] if large else variable_numeric_params_sorted)

    # History
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.update_layout(width=900, height=360, margin=dict(l=60, r=20, t=40, b=60))
        pio.write_html(fig, out_dir / f"history-{study_name}.html", include_plotlyjs=True)
        try:
            fig.write_image(out_dir / f"history-{study_name}.png", width=900, height=360, scale=2)
        except Exception:
            pass
    except Exception:
        pass

    # Slice
    try:
        categorical_params = [p for p in variable_params_sorted if p not in numeric_params]
        long_cat: set[str] = set()
        for p in categorical_params:
            try:
                vals = [str(v) for v in all_params_values.get(p, [])]
                if vals and max(len(s) for s in vals) > 40:
                    long_cat.add(p)
            except Exception:
                pass

        candidates = [p for p in params_slice if p not in long_cat]
        max_dims = cfg.slice_dims_large if cfg.light_mode else max(12, cfg.slice_dims_large)
        params_for_slice = candidates[:max_dims] if candidates else variable_params_sorted[:max_dims]

        fig = optuna.visualization.plot_slice(study, params=params_for_slice)
        facet_count = max(1, len(params_for_slice))
        width = max(1400, min(3800, 240 * facet_count))
        fig.update_layout(width=width, height=760, margin=dict(l=100, r=40, t=60, b=140))
        try:
            for ax_name in [k for k in fig.layout if str(k).startswith("xaxis")]:
                fig.layout[ax_name].update(tickangle=30, automargin=True)
            for ay_name in [k for k in fig.layout if str(k).startswith("yaxis")]:
                fig.layout[ay_name].update(automargin=True)
        except Exception:
            pass
        pio.write_html(fig, out_dir / f"slice-{study_name}.html", include_plotlyjs=True)
        try:
            fig.write_image(out_dir / f"slice-{study_name}.png", width=width, height=760, scale=2)
        except Exception:
            pass
    except Exception:
        pass

    # Contour (numeric only)
    try:
        params_contour_numeric = params_contour or variable_numeric_params_sorted
        if params_contour_numeric:
            fig = optuna.visualization.plot_contour(study, params=params_contour_numeric)
            fig.update_layout(width=1200, height=900, margin=dict(l=80, r=40, t=50, b=100))
            fig.update_xaxes(tickangle=45, automargin=True)
            fig.update_yaxes(automargin=True)
            pio.write_html(fig, out_dir / f"contour-{study_name}.html", include_plotlyjs=True)
            try:
                fig.write_image(out_dir / f"contour-{study_name}.png", width=1200, height=900, scale=2)
            except Exception:
                pass
    except Exception:
        pass

    # Importances (variable params only)
    try:
        imp_params = variable_params_sorted if variable_params_sorted else sorted_params
        fig = optuna.visualization.plot_param_importances(study, params=imp_params)
        fig.update_layout(width=1100, height=620, margin=dict(l=60, r=20, t=40, b=60))
        pio.write_html(fig, out_dir / f"importances-{study_name}.html", include_plotlyjs=True)
        try:
            fig.write_image(out_dir / f"importances-{study_name}.png", width=1100, height=620, scale=2)
        except Exception:
            pass
    except Exception:
        pass

    # Parallel coordinates
    try:
        par_candidates = params_parallel or variable_params_sorted or sorted_params
        if n_complete >= cfg.large_trials_threshold and cfg.light_mode:
            par_params = list(par_candidates)[:cfg.parallel_dims_large]
        else:
            par_params = list(par_candidates)

        fig = optuna.visualization.plot_parallel_coordinate(study, params=par_params)
        try:
            for tr in fig.data:
                if getattr(tr, "type", None) == "parcoords":
                    try:
                        tr.line.colorscale = "Turbo"
                        tr.line.showscale = True
                        tr.line.width = 2.2
                        tr.opacity = 0.9
                    except Exception:
                        pass
                    try:
                        tr.labelfont = dict(size=14)
                        tr.tickfont = dict(size=12)
                    except Exception:
                        pass
            fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        except Exception:
            pass

        num_axes = max(1, len(par_params) + 1)
        width_par = max(1600, min(5200, 260 * num_axes))
        fig.update_layout(width=width_par, height=720, margin=dict(l=60, r=20, t=40, b=120))
        pio.write_html(fig, out_dir / f"parallel-{study_name}.html", include_plotlyjs=True)
        if not (cfg.light_mode and n_complete >= cfg.large_trials_threshold and cfg.skip_png_on_large and not cfg.force_png_parallel):
            try:
                fig.write_image(out_dir / f"parallel-{study_name}.png", width=width_par, height=720, scale=4)
            except Exception:
                pass
    except Exception:
        pass

    return True

def write_fallback_history_from_csv(study_name: str, df: pd.DataFrame, out_dir: Path, png_scale: int) -> None:
    try:
        import plotly.express as px
        out_dir.mkdir(exist_ok=True)
        x = df["trial_id"] if "trial_id" in df.columns else df.index
        fig = px.scatter(df, x=x, y="mean_acc", title=f"Mean accuracy per trial · {study_name}")
        pio.write_html(fig, out_dir / f"history_basic-{study_name}.html", include_plotlyjs=True)
        try:
            fig.write_image(out_dir / f"history_basic-{study_name}.png", scale=png_scale)
        except Exception:
            pass
        print(f"  - wrote fallback plots in {out_dir.name}")
    except Exception:
        pass
