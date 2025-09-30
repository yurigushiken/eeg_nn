# optuna_tools/reports.py
from __future__ import annotations

import base64
from pathlib import Path
import pandas as pd


def _plots_dir(study_dir: Path) -> Path:
    return study_dir / f"!plots-{study_dir.name}"


def _embed_image_as_data_uri(png_path: Path) -> str:
    try:
        with open(png_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def _resolve_confusion_png(trial_dir: Path) -> Path:
    """Search for a confusion-matrix PNG in new/legacy layouts."""
    plots_dir_new = trial_dir / "plots_outer"
    plots_dir_legacy = trial_dir / "plots"

    primary_new = plots_dir_new / "overall_confusion.png"
    if primary_new.exists():
        return primary_new

    primary_legacy_dir = plots_dir_legacy / "overall_confusion.png"
    if primary_legacy_dir.exists():
        return primary_legacy_dir

    primary_legacy = trial_dir / "overall_confusion.png"
    if primary_legacy.exists():
        return primary_legacy

    alts_new = sorted(plots_dir_new.glob("overall*_confusion*.png")) if plots_dir_new.exists() else []
    if alts_new:
        return alts_new[0]
    alts_legacy_dir = sorted(plots_dir_legacy.glob("overall*_confusion*.png")) if plots_dir_legacy.exists() else []
    if alts_legacy_dir:
        return alts_legacy_dir[0]
    alts_legacy = sorted(trial_dir.glob("overall*_confusion*.png"))
    if alts_legacy:
        return alts_legacy[0]

    fold_new = sorted(plots_dir_new.glob("fold*_confusion.png")) if plots_dir_new.exists() else []
    if fold_new:
        return fold_new[0]
    fold_legacy_dir = sorted(plots_dir_legacy.glob("fold*_confusion.png")) if plots_dir_legacy.exists() else []
    if fold_legacy_dir:
        return fold_legacy_dir[0]
    fold_legacy = sorted(trial_dir.glob("fold*_confusion.png"))
    if fold_legacy:
        return fold_legacy[0]

    return primary_new  # may not exist; handled by caller


def write_top3_report(study_dir: Path, df: pd.DataFrame, study_name: str) -> None:
    top = df.head(3).copy()
    if top.empty:
        return

    entries = []
    for _, row in top.iterrows():
        trial_dir_name = str(row.get("trial_dir") or row.get("run_id") or "")
        if not trial_dir_name:
            continue
        trial_dir = study_dir / trial_dir_name
        cm_path = _resolve_confusion_png(trial_dir)
        data_uri = _embed_image_as_data_uri(cm_path)

        acc = row.get("mean_acc")
        inner_f1 = row.get("inner_mean_macro_f1")
        inner_acc = row.get("inner_mean_acc")
        bits = []
        if pd.notna(acc):
            bits.append(f"acc={float(acc):.2f}%")
        if pd.notna(inner_f1):
            bits.append(f"inner F1={float(inner_f1):.2f}")
        if pd.notna(inner_acc):
            bits.append(f"inner acc={float(inner_acc):.2f}%")
        for k in ("lr", "batch_size"):
            v = row.get(k)
            if pd.notna(v):
                bits.append(f"{k}={v}")
        caption = f"tid={row.get('trial_id','?')} · " + ", ".join(bits)

        entries.append({"trial_dir_name": trial_dir_name, "data_uri": data_uri, "caption": caption, "cm_path": str(cm_path)})

    if not entries:
        return

    # Gather Optuna plots (embed as base64) and keep file paths for PDF fallback
    history_card = None
    other_cards = []
    plot_paths: list[str] = []
    try:
        plots_dir = _plots_dir(study_dir)
        if plots_dir.exists():
            hist_path: str | None = None
            other_paths: list[str] = []
            for png in sorted(plots_dir.glob("*.png")):
                uri = _embed_image_as_data_uri(png)
                if uri:
                    if png.name.startswith("history-") or png.name.startswith("history_basic-"):
                        history_card = {"name": png.name, "uri": uri}
                        hist_path = str(png)
                    else:
                        other_cards.append({"name": png.name, "uri": uri})
                        other_paths.append(str(png))
            plot_paths = ([hist_path] if hist_path else []) + other_paths
    except Exception:
        pass

    title = f"Top 3 Trials – {study_name}"
    plots_section_html = ""
    if history_card or other_cards:
        history_block = f"""
    <div class="plot-history">
      <img src="{history_card['uri']}" alt="{history_card['name']}"> 
      <div class="caption">{history_card['name']}</div>
    </div>
        """ if history_card else ""
        stacked_items = "".join([
            f'<div class="plot-history">\n  <img src="{c["uri"]}" alt="{c["name"]}">\n  <div class="caption">{c["name"]}</div>\n</div>'
            for c in other_cards
        ])
        plots_section_html = f"""
  <div class="plots-section">
    <h2>Optuna Study Plots</h2>
    {history_block}
    {stacked_items}
  </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 18px; }}
    h1 {{ text-align: center; margin-bottom: 8px; }}
    .subtitle {{ text-align: center; color: #555; margin-bottom: 14px; }}

    .hero {{ display: flex; justify-content: center; margin-bottom: 10px; }}
    .hero img {{ max-width: 85%; height: auto; border: 1px solid #ccc; }}
    .hero .caption {{ max-width: 85%; font-size: 12px; color: #333; margin: 6px auto 0; text-align: center; }}

    .row {{ display: flex; gap: 8px; justify-content: center; align-items: flex-start; }}
    .row .item {{ flex: 1; max-width: 46%; text-align: center; }}
    .row img {{ width: 100%; height: auto; border: 1px solid #ccc; }}
    .caption {{ font-size: 12px; color: #333; margin-top: 6px; }}

    .plots-section {{ margin-top: 16px; }}

    .plot-history {{ text-align: center; margin: 6px 0 12px 0; }}
    .plot-history img {{ width: 65%; height: auto; border: 1px solid #ccc; }}

    @page {{ size: A4; margin: 12mm; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="subtitle">Confusion matrices for the top three trials (best by inner macro-F1/acc)</div>

  <div class="hero">
    <div>
      <img src="{entries[0]['data_uri']}" alt="Top 1 confusion matrix" />
      <div class="caption">{entries[0]['caption']}</div>
    </div>
  </div>

  <div class="row">
    {''.join([f'<div class="item"><img src="{e["data_uri"]}" alt="Top confusion" /><div class="caption">{e["caption"]}</div></div>' for e in entries[1:3]])}
  </div>

  {plots_section_html}
</body>
</html>
"""

    html_out = study_dir / f"!top3-report-{study_name}.html"
    html_out.write_text(html, encoding="utf-8")
    print(f"  - wrote top3 HTML: {html_out.name}")

    # PDF via Playwright → fallback to matplotlib
    try:
        from playwright.sync_api import sync_playwright
        pdf_out = study_dir / f"!top3-report-{study_name}.pdf"
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{html_out.resolve()}")
            page.pdf(path=str(pdf_out), format='A4', print_background=True,
                     margin={'top': '12mm', 'bottom': '12mm', 'left': '12mm', 'right': '12mm'})
            browser.close()
        print(f"  - wrote top3 PDF: {pdf_out.name}")
    except Exception:
        try:
            import textwrap
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.image as mpimg

            pdf_out = study_dir / f"!top3-report-{study_name}.pdf"
            with PdfPages(pdf_out) as pdf:
                fig_w, fig_h = 8.27, 11.69
                fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

                has_history = bool(plot_paths and plot_paths[0] and (
                    "history-" in Path(plot_paths[0]).name or "history_basic-" in Path(plot_paths[0]).name))
                other_paths = plot_paths[1:] if has_history else plot_paths

                # hero 3.2, small conf row 2.6, history 2.6, each other row 2.6
                other_rows = len(other_paths) if other_paths else 0
                total_rows = (3 + other_rows) if has_history else (2 + other_rows)
                heights = [3.2, 2.6] + ([2.6] if has_history else []) + [2.6] * other_rows

                gs = fig.add_gridspec(ncols=2, nrows=total_rows, height_ratios=heights, hspace=0.32, wspace=0.12)
                fig.suptitle(f"Top 3 Trials – {study_name}", fontsize=12)

                ax_hero = fig.add_subplot(gs[0, :])
                try:
                    img_hero = mpimg.imread(entries[0]["cm_path"])
                    ax_hero.imshow(img_hero, interpolation="none")
                except Exception:
                    ax_hero.text(0.5, 0.5, "[missing hero image]", ha='center', va='center')
                ax_hero.axis('off')
                hero_caption = textwrap.fill(entries[0]["caption"], width=80)
                ax_hero.set_title(hero_caption, fontsize=7, pad=2)

                for idx in range(2):
                    ax = fig.add_subplot(gs[1, idx])
                    if len(entries) > 1 + idx:
                        try:
                            img = mpimg.imread(entries[1 + idx]["cm_path"])
                            ax.imshow(img, interpolation="none")
                        except Exception:
                            ax.text(0.5, 0.5, "[missing image]", ha='center', va='center')
                        small_caption = textwrap.fill(entries[1 + idx]["caption"], width=50)
                        ax.set_title(small_caption, fontsize=7, pad=2)
                    ax.axis('off')

                start_row = 2
                if has_history:
                    ax_hist = fig.add_subplot(gs[start_row, :])
                    try:
                        img_h = mpimg.imread(plot_paths[0])
                        ax_hist.imshow(img_h, interpolation="none")
                    except Exception:
                        ax_hist.text(0.5, 0.5, "[missing history]", ha='center', va='center')
                    ax_hist.axis('off')
                    ax_hist.set_title(Path(plot_paths[0]).name, fontsize=7, pad=2)
                    start_row += 1

                for i, pth in enumerate(other_paths):
                    r = start_row + i
                    ax = fig.add_subplot(gs[r, :])
                    try:
                        img = mpimg.imread(pth)
                        ax.imshow(img, interpolation="none")
                    except Exception:
                        ax.text(0.5, 0.5, "[missing plot]", ha='center', va='center')
                    ax.axis('off')
                    ax.set_title(Path(pth).name, fontsize=7, pad=2)

                fig.subplots_adjust(top=0.92, bottom=0.06, left=0.06, right=0.94)
                pdf.savefig(fig)
                plt.close(fig)
            print(f"  - wrote top3 PDF via matplotlib fallback: {pdf_out.name}")
        except Exception as e2:
            print(f"  - skipping top3 PDF (fallback failed): {e2}")
