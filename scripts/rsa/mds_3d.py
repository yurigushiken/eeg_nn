from __future__ import annotations

from pathlib import Path

import pandas as pd


def camera_eye_from_azim_elev(*, azim_deg: float, elev_deg: float, radius: float = 2.0) -> dict:
    """
    Convert (azimuth, elevation) in degrees into a Plotly 3D camera eye vector.

    This mirrors the common Matplotlib mental model (azim/elev) while producing
    Plotly's camera.eye dict. Angles:
      - azim: rotation around z-axis (0° = +x, 90° = +y)
      - elev: angle above the xy plane
    """
    import math

    az = math.radians(float(azim_deg))
    el = math.radians(float(elev_deg))
    r = float(radius)
    x = r * math.cos(az) * math.cos(el)
    y = r * math.sin(az) * math.cos(el)
    z = r * math.sin(el)
    return dict(x=x, y=y, z=z)


def _theme_layout(theme: str) -> dict:
    t = (theme or "").strip().lower()
    if t not in {"dark", "light"}:
        raise ValueError(f"theme must be 'dark' or 'light', got {theme!r}")

    if t == "dark":
        template = "plotly_dark"
        bg = "rgb(15,15,15)"
        fg = "rgb(235,235,235)"
        grid = "rgba(255,255,255,0.18)"
    else:
        template = "plotly_white"
        bg = "rgb(255,255,255)"
        fg = "rgb(20,20,20)"
        grid = "rgba(0,0,0,0.15)"

    axis = dict(
        showbackground=True,
        backgroundcolor=bg,
        gridcolor=grid,
        zerolinecolor=grid,
        color=fg,
        title=dict(font=dict(color=fg)),
        tickfont=dict(color=fg),
    )

    return dict(
        template=template,
        paper_bgcolor=bg,
        font=dict(color=fg),
        scene=dict(
            bgcolor=bg,
            xaxis=axis,
            yaxis=axis,
            zaxis=axis,
            aspectmode="data",
        ),
    )


def plot_mds_3d_html(
    positions_3d: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    theme: str = "dark",
) -> None:
    """
    Write an interactive 3D MDS scatter to a self-contained HTML file.

    Intended for smooth local exploration (rotate/zoom/pan) with a built-in
    dark/light theme toggle.
    """
    required_cols = {"label", "x", "y", "z"}
    missing = required_cols.difference(set(positions_3d.columns))
    if missing:
        raise ValueError(f"positions_3d is missing required columns: {sorted(missing)}")

    # Lazy import so the rest of the pipeline does not require Plotly unless requested.
    import plotly.graph_objects as go
    import plotly.io as pio

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [str(v) for v in positions_3d["label"].tolist()]
    x = positions_3d["x"].to_numpy()
    y = positions_3d["y"].to_numpy()
    z = positions_3d["z"].to_numpy()

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                text=labels,
                textposition="top center",
                marker=dict(
                    size=8,
                    color="#2a6f97",
                    opacity=0.9,
                    line=dict(width=0),
                ),
                hovertemplate="Label %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        ]
    )

    # Deterministic camera for reproducibility (especially important for screenshots).
    camera = dict(eye=dict(x=1.25, y=1.25, z=1.25))

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, b=0, t=55),
        scene_camera=camera,
        **_theme_layout(theme),
    )

    # Built-in dark/light toggle (no external UI framework needed).
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
                buttons=[
                    dict(label="Light", method="relayout", args=[_theme_layout("light")]),
                    dict(label="Dark", method="relayout", args=[_theme_layout("dark")]),
                ],
            )
        ]
    )

    pio.write_html(
        fig,
        file=str(output_path),
        include_plotlyjs=True,
        full_html=True,
        auto_play=False,
    )


def plot_mds_3d_static_views(
    positions_3d: pd.DataFrame,
    output_dir: Path,
    *,
    basename: str,
    title: str,
    theme: str = "dark",
    projection: str = "perspective",
    views: list[tuple[str, float, float]] | None = None,
) -> list[Path]:
    """
    Export a small set of fixed-angle static PNGs for publication selection.

    Returns list of written file paths.
    """
    required_cols = {"label", "x", "y", "z"}
    missing = required_cols.difference(set(positions_3d.columns))
    if missing:
        raise ValueError(f"positions_3d is missing required columns: {sorted(missing)}")

    proj = (projection or "").strip().lower()
    if proj not in {"perspective", "orthographic"}:
        raise ValueError(f"projection must be 'perspective' or 'orthographic', got {projection!r}")

    # Default: four rotations at a single elevation (easy to compare and pick).
    if views is None:
        views = [
            ("az045_el30", 45.0, 30.0),
            ("az135_el30", 135.0, 30.0),
            ("az225_el30", 225.0, 30.0),
            ("az315_el30", 315.0, 30.0),
        ]

    import plotly.graph_objects as go

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [str(v) for v in positions_3d["label"].tolist()]
    x = positions_3d["x"].to_numpy()
    y = positions_3d["y"].to_numpy()
    z = positions_3d["z"].to_numpy()

    written: list[Path] = []
    for view_id, azim_deg, elev_deg in views:
        eye = camera_eye_from_azim_elev(azim_deg=azim_deg, elev_deg=elev_deg, radius=2.0)
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers+text",
                    text=labels,
                    textposition="top center",
                    marker=dict(size=8, color="#2a6f97", opacity=0.95),
                    hovertemplate="Label %{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=title,
            margin=dict(l=0, r=0, b=0, t=55),
            scene_camera=dict(
                eye=eye,
                projection=dict(type=proj),
            ),
            **_theme_layout(theme),
        )

        out_path = output_dir / f"{basename}__{view_id}.png"
        try:
            # Requires kaleido (present in eegnex-env). If missing, this will raise.
            fig.write_image(str(out_path), width=1200, height=900, scale=2)
        except Exception as e:
            raise RuntimeError(
                "Static export failed. This typically means the Plotly image exporter isn't available. "
                "Install Kaleido in your environment (e.g., `pip install -U kaleido`) and retry."
            ) from e
        written.append(out_path)

    return written


