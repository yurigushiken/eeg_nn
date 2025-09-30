# optuna_tools/db.py
from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_optuna_db(optuna_db_dir: Path, study_name: str) -> Optional[Path]:
    cand = optuna_db_dir / f"{study_name}.db"
    return cand if cand.exists() else None


def generate_plots_from_db(
    *,
    target_study_name: str,
    db_path: Path,
    out_dir: Path,
    light_mode: bool,
    n_trials_large: int,
    dims_parallel_large: int,
    dims_slice_large: int,
    dims_contour_large: int,
    skip_png_on_large: bool,
    force_png_parallel: bool,
    force_png_contour: bool,
    png_scale: int,
) -> bool:
    """
    Port of your improved plotting from the latest monolith, parameterized via Config.
    Returns True if at least one Optuna plot (not fallback) was generated successfully.
    """
    import optuna
    import plotly.io as pio

    try:
        study = optuna.load_study(study_name=target_study_name, storage=f"sqlite:///{db_path}")
    except Exception:
        return False

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not trials:
        return False

    out_dir.mkdir(exist_ok=True)

    # ==== collect param stats ====
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

    # Exclude constant-valued params; keep only those with >1 unique value
    try:
        unique_counts: dict[str, int] = {k: len(set(vs)) for k, vs in all_params_values.items()}
    except Exception:
        unique_counts = {k: 2 for k in sorted_params}
    variable_params_sorted = [k for k in sorted_params if unique_counts.get(k, 0) > 1]
    variable_numeric_params_sorted = [k for k in variable_params_sorted if k in numeric_params]
    if not variable_params_sorted:
        variable_params_sorted = list(sorted_params)

    n_complete = len(trials)
    large = light_mode and (n_complete >= n_trials_large)

    params_parallel = (variable_params_sorted[:dims_parallel_large] if large else variable_params_sorted)
    params_slice = (variable_params_sorted[:dims_slice_large] if large else variable_params_sorted)
    params_contour = (variable_numeric_params_sorted[:dims_contour_large] if large else variable_numeric_params_sorted)

    # ---- history ----
    any_ok = False
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.update_layout(width=900, height=360, margin=dict(l=60, r=20, t=40, b=60))
        pio.write_html(fig, out_dir / f"history-{target_study_name}.html", include_plotlyjs=True)
        try:
            fig.write_image(out_dir / f"history-{target_study_name}.png", width=900, height=360, scale=2)
        except Exception:
            pass
        any_ok = True
    except Exception:
        pass

    # ---- slice ----
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
        max_dims = dims_slice_large if light_mode else max(12, dims_slice_large)
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
        pio.write_html(fig, out_dir / f"slice-{target_study_name}.html", include_plotlyjs=True)
        try:
            fig.write_image(out_dir / f"slice-{target_study_name}.png", width=width, height=760, scale=2)
        except Exception:
            pass
        any_ok = True
    except Exception:
        pass

    # ---- contour (numeric only) ----
    try:
        params_contour_numeric = params_contour or variable_numeric_params_sorted
        if params_contour_numeric:
            fig = optuna.visualization.plot_contour(study, params=params_contour_numeric)
            fig.update_layout(width=1200, height=900, margin=dict(l=80, r=40, t=50, b=100))
            fig.update_xaxes(tickangle=45, automargin=True)
            fig.update_yaxes(automargin=True)
            pio.write_html(fig, out_dir / f"contour-{target_study_name}.html", include_plotlyjs=True)
            try:
                fig.write_image(out_dir / f"contour-{target_study_name}.png", width=1200, height=900, scale=2)
            except Exception:
                pass
            any_ok = True
    except Exception:
        pass

    # ---- importances (variable params only) ----
    try:
        imp_params = variable_params_sorted if variable_params_sorted else sorted_params
        fig = optuna.visualization.plot_param_importances(study, params=imp_params)
        fig.update_layout(width=1100, height=620, margin=dict(l=60, r=20, t=40, b=60))
        pio.write_html(fig, out_dir / f"importances-{target_study_name}.html", include_plotlyjs=True)
        try:
            fig.write_image(out_dir / f"importances-{target_study_name}.png", width=1100, height=620, scale=2)
        except Exception:
            pass
        any_ok = True
    except Exception:
        pass

    # ---- parallel coordinate ----
    try:
        par_candidates = params_parallel or variable_params_sorted or sorted_params
        if large:
            par_params = list(par_candidates)[:dims_parallel_large]
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
        pio.write_html(fig, out_dir / f"parallel-{target_study_name}.html", include_plotlyjs=True)
        if not (large and skip_png_on_large and not force_png_parallel):
            try:
                fig.write_image(out_dir / f"parallel-{target_study_name}.png", width=width_par, height=720, scale=4)
            except Exception:
                pass
        any_ok = True
    except Exception:
        pass

    return any_ok
