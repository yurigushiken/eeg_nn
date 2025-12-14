"""
Generate publication-ready tables for RSA matrix results.

Creates LaTeX, CSV, and PNG versions of statistical tables comparing
pairwise decoding accuracies with p-values and significance markers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.table import Table

# Ensure project root is importable when running as a script.
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from scripts.rsa.naming import analysis_id_from_run_root, prefixed_title

def load_rsa_data(csv_path: Path) -> pd.DataFrame:
    """Load RSA results CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def infer_code_style(df: pd.DataFrame) -> str:
    """
    Infer how ClassA/ClassB are encoded.

    Returns:
        - "digit": ClassA/B are already numerosities (1–6), e.g., landing-digit stats.
        - "cardinality": ClassA/B are 11,22,... (same-digit cardinality codes).
        - "raw": mixed/other; treat ClassA/B as-is.
    """
    codes = set(pd.concat([df["ClassA"], df["ClassB"]]).unique())
    if all(c <= 10 for c in codes):
        return "digit"
    if all(c % 11 == 0 for c in codes):
        return "cardinality"
    return "raw"


def condition_code_to_numerosity(code: int, style: str) -> int:
    """
    Convert condition code to numerosity based on encoding style.
    """
    if style == "digit":
        return code
    if style == "cardinality":
        return code // 11
    # raw: pass through (used when comparisons are given directly in code space)
    return code


def extract_comparison_data(
    df: pd.DataFrame,
    comparisons: List[Tuple[int, int]],
    metric: str = "mean_accuracy",
    code_style: str = "auto",
) -> pd.DataFrame:
    """
    Extract accuracy and p-value for specified numerosity comparisons.

    Args:
        df: RSA stats summary dataframe (with ClassA, ClassB as condition codes)
        comparisons: List of (numerosity_a, numerosity_b) tuples (e.g., (1, 2))
        metric: Metric column to extract (default: mean_accuracy)

    Returns:
        DataFrame with columns: Comparison, Accuracy, p_value, Significance
    """
    rows = []
    style = code_style
    if style == "auto":
        style = infer_code_style(df)

    for a, b in comparisons:
        if style == "cardinality":
            # Convert numerosities to condition codes (1→11, 2→22, etc.)
            code_a = a * 11
            code_b = b * 11
        else:
            # digit/raw: treat comparisons as already in the same space as ClassA/B
            code_a = a
            code_b = b

        # Find matching row (order-independent)
        mask = ((df["ClassA"] == code_a) & (df["ClassB"] == code_b)) | (
            (df["ClassA"] == code_b) & (df["ClassB"] == code_a)
        )
        subset = df[mask]

        if subset.empty:
            print(f"Warning: No data found for {a}v{b} (codes {code_a}v{code_b})")
            continue

        # Use first row (should be unique per comparison)
        row_data = subset.iloc[0]
        acc = row_data[metric]

        # Get p-value
        pval = np.nan
        for candidate in ("p_value_holm", "p_value", "pval"):
            if candidate in row_data:
                pval = row_data[candidate]
                break

        # Significance markers
        if pd.notna(pval):
            if pval < 0.001:
                sig = "***"
            elif pval < 0.01:
                sig = "**"
            elif pval < 0.05:
                sig = "*"
            else:
                sig = ""
        else:
            sig = ""

        rows.append(
            {
                "Comparison": f"{a}v{b}",
                "Accuracy": acc,
                "p_value": pval,
                "Significance": sig,
            }
        )

    return pd.DataFrame(rows)


def format_latex_table(
    data: pd.DataFrame,
    title: str,
    caption: str = "",
    label: str = "tab:rsa",
) -> str:
    """
    Generate publication-ready LaTeX table using booktabs style.

    Args:
        data: DataFrame with Comparison, Accuracy, p_value, Significance
        title: Table title
        caption: LaTeX caption (optional)
        label: LaTeX label for referencing (optional)

    Returns:
        LaTeX table string
    """
    # Format accuracy to 1 decimal, p-values to scientific notation
    data_formatted = data.copy()
    data_formatted["Accuracy"] = data_formatted["Accuracy"].apply(lambda x: f"{x:.1f}")

    def format_pval(p):
        if pd.isna(p):
            return "-"
        elif p < 0.001:
            return "<0.001"
        else:
            return f"{p:.3f}"

    data_formatted["p_value"] = data_formatted["p_value"].apply(format_pval)

    # Build LaTeX manually for full control (booktabs style)
    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "  \\begin{tabular}{lccc}",
        "    \\toprule",
        "    Comparison & Accuracy (\\%) & \\textit{p}-value & Sig. \\\\",
        "    \\midrule",
    ]

    for _, row in data_formatted.iterrows():
        comp = row["Comparison"]
        acc = row["Accuracy"]
        pval = row["p_value"]
        sig = row["Significance"]
        lines.append(f"    {comp} & {acc} & {pval} & {sig} \\\\")

    lines.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
        ]
    )

    return "\n".join(lines)


def plot_table_as_image(
    data: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    """
    Render table as PNG image using matplotlib.

    Args:
        data: DataFrame with Comparison, Accuracy, p_value, Significance
        title: Table title
        output_path: Output PNG path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Format data
    data_formatted = data.copy()
    data_formatted["Accuracy"] = data_formatted["Accuracy"].apply(lambda x: f"{x:.1f}")

    def format_pval(p):
        if pd.isna(p):
            return "-"
        elif p < 0.001:
            return "<0.001"
        else:
            return f"{p:.3f}"

    data_formatted["p_value"] = data_formatted["p_value"].apply(format_pval)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, len(data_formatted) * 0.4 + 1.5))
    ax.axis("off")

    # Column headers
    col_labels = ["Comparison", "Accuracy (%)", "p-value", "Sig."]
    cell_text = data_formatted.values.tolist()

    # Create table
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.25, 0.25, 0.25, 0.25],
    )

    # Style table (R-like aesthetics)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Header styling
    for i in range(len(col_labels)):
        cell = table[(0, i)]
        cell.set_facecolor("#E8E8E8")
        cell.set_text_props(weight="bold")

    # Alternating row colors (subtle)
    for i in range(1, len(cell_text) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor("#F9F9F9")

    ax.set_title(title, fontsize=12, weight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_table_set(
    df: pd.DataFrame,
    comparisons: List[Tuple[int, int]],
    title: str,
    output_prefix: str,
    output_dir: Path,
    caption: str = "",
    label: str = "tab:rsa",
    code_style: str = "auto",
) -> None:
    """
    Generate LaTeX, CSV, and PNG for one table.

    Args:
        df: RSA results dataframe
        comparisons: List of (classA, classB) tuples
        title: Table title
        output_prefix: Filename prefix (e.g., "table1_untitled")
        output_dir: Output directory
        caption: LaTeX caption
        label: LaTeX label
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    data = extract_comparison_data(df, comparisons, code_style=code_style)

    if data.empty:
        print(f"Warning: No data extracted for {title}")
        return

    # Generate LaTeX
    latex_str = format_latex_table(data, title, caption, label)
    latex_path = output_dir / f"{output_prefix}.tex"
    latex_path.write_text(latex_str, encoding="utf-8")
    print(f"[generate_rsa_tables] LaTeX saved to {latex_path}")

    # Generate CSV
    csv_path = output_dir / f"{output_prefix}.csv"
    data.to_csv(csv_path, index=False)
    print(f"[generate_rsa_tables] CSV saved to {csv_path}")

    # Generate PNG
    png_path = output_dir / f"{output_prefix}.png"
    plot_table_as_image(data, title, png_path)
    print(f"[generate_rsa_tables] PNG saved to {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-ready RSA tables.")
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to stats_summary.csv (output from analyze_rsa_stats.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save tables (e.g., results/runs/rsa_matrix_v1/tables).",
    )
    parser.add_argument(
        "--metric",
        default="Accuracy",
        help="Metric column to use (default: Accuracy).",
    )
    parser.add_argument(
        "--code-style",
        choices=["auto", "digit", "cardinality", "raw"],
        default="auto",
        help="How ClassA/ClassB are encoded: auto-detect, raw, digit (1-6), or cardinality (11,22,...).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    df = load_rsa_data(args.csv)

    # Convention: output_dir is typically <run_root>/tables.
    run_root = args.output_dir.parent if args.output_dir.name == "tables" else args.output_dir
    analysis_id = analysis_id_from_run_root(run_root)

    # Table 1: Untitled (PI/ANS boundary comparisons)
    table1_comparisons = [
        (1, 2),
        (2, 3),
        (3, 4),
        (3, 6),
        (4, 5),
        (5, 6),
    ]
    generate_table_set(
        df=df,
        comparisons=table1_comparisons,
        title=prefixed_title(run_root=run_root, title="Untitled"),
        output_prefix=f"{analysis_id}__table1_untitled",
        output_dir=args.output_dir,
        caption="Pairwise decoding accuracies for numerosity comparisons.",
        label="tab:rsa_untitled",
        code_style=args.code_style,
    )

    # Table 2: One vs. All (decoding 1 against higher numerosities)
    table2_comparisons = [
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (1, 6),
    ]
    generate_table_set(
        df=df,
        comparisons=table2_comparisons,
        title=prefixed_title(run_root=run_root, title="Decoding One Against Higher Numerosities"),
        output_prefix=f"{analysis_id}__table2_one_vs_all",
        output_dir=args.output_dir,
        caption="Decoding accuracy for contrasts between numerosity 1 and all higher numerosities (2–6).",
        label="tab:rsa_one_vs_all",
        code_style=args.code_style,
    )

    # Table 3: All Pairwise Comparisons (complete RSA matrix)
    table3_comparisons = [
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
        (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 4), (3, 5), (3, 6),
        (4, 5), (4, 6),
        (5, 6),
    ]
    generate_table_set(
        df=df,
        comparisons=table3_comparisons,
        title=prefixed_title(run_root=run_root, title="Complete Pairwise Decoding Accuracy Matrix"),
        output_prefix=f"{analysis_id}__table3_all_pairs",
        output_dir=args.output_dir,
        caption="Decoding accuracies for all pairwise numerosity comparisons (1–6).",
        label="tab:rsa_all_pairs",
        code_style=args.code_style,
    )

    print(f"\n[generate_rsa_tables] All tables saved to {args.output_dir}")


if __name__ == "__main__":
    main()
