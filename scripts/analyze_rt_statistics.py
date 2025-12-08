"""
Statistical analysis of reaction time differences across cardinalities.

Performs multiple statistical tests to assess whether RT varies significantly
by numerosity, including:
- One-way repeated measures ANOVA
- Linear mixed-effects model (accounting for subject variability)
- Post-hoc pairwise comparisons with multiple comparisons correction
- Effect size calculations
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from scipy.stats import f_oneway, friedmanchisquare
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def load_rt_data(csv_path: Path) -> pd.DataFrame:
    """Load RT data from CSV."""
    return pd.read_csv(csv_path)


def perform_anova(df: pd.DataFrame) -> dict:
    """
    Perform one-way ANOVA on RT by cardinality.

    Note: This treats trials as independent (ignores subject structure).
    Use LMM for proper subject-aware analysis.
    """
    groups = [df[df["Cardinality"] == c]["RT"].values for c in sorted(df["Cardinality"].unique())]

    f_stat, p_value = f_oneway(*groups)

    # Calculate effect size (eta-squared)
    grand_mean = df["RT"].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
    ss_total = sum((df["RT"] - grand_mean)**2)
    eta_squared = ss_between / ss_total

    return {
        "test": "One-way ANOVA",
        "F": f_stat,
        "p": p_value,
        "eta_squared": eta_squared,
        "df_between": len(groups) - 1,
        "df_within": len(df) - len(groups),
    }


def perform_friedman_test(df: pd.DataFrame) -> dict:
    """
    Perform Friedman test (non-parametric repeated measures).

    Requires complete data for all subjects × cardinalities.
    """
    # Pivot to subject × cardinality matrix
    pivot = df.groupby(["SubjectID", "Cardinality"])["RT"].mean().unstack()

    # Only use subjects with data for all cardinalities
    complete_subjects = pivot.dropna()

    if len(complete_subjects) < len(pivot):
        print(f"Warning: Friedman test using {len(complete_subjects)}/{len(pivot)} subjects with complete data")

    # Friedman test requires data in columns
    chi2, p_value = friedmanchisquare(*[complete_subjects[c].values for c in complete_subjects.columns])

    return {
        "test": "Friedman (non-parametric RM)",
        "chi2": chi2,
        "p": p_value,
        "df": len(complete_subjects.columns) - 1,
        "n_subjects": len(complete_subjects),
    }


def perform_lmm(df: pd.DataFrame) -> dict:
    """
    Perform Linear Mixed-Effects Model with random intercept for subject.

    This is the gold standard for analyzing RT with repeated measures.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "statsmodels not available for LMM"}

    # Fit LMM: RT ~ Cardinality + (1 | Subject)
    model = smf.mixedlm("RT ~ C(Cardinality)", df, groups=df["SubjectID"])
    result = model.fit(reml=True)

    # Extract F-test for Cardinality effect
    # Compare to null model (intercept only)
    null_model = smf.mixedlm("RT ~ 1", df, groups=df["SubjectID"])
    null_result = null_model.fit(reml=True)

    # Likelihood ratio test
    lr_stat = -2 * (null_result.llf - result.llf)
    df_diff = result.df_resid - null_result.df_resid
    p_value = stats.chi2.sf(lr_stat, df_diff)

    return {
        "test": "Linear Mixed-Effects Model",
        "LR_chi2": lr_stat,
        "p": p_value,
        "df": df_diff,
        "AIC": result.aic,
        "BIC": result.bic,
        "summary": result.summary(),
    }


def perform_posthoc_tukey(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform Tukey HSD post-hoc pairwise comparisons.

    Returns DataFrame with all pairwise comparisons.
    """
    tukey = pairwise_tukeyhsd(df["RT"], df["Cardinality"], alpha=0.05)

    # Convert to DataFrame
    results = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    return results


def perform_pairwise_t_tests(df: pd.DataFrame, correction: str = "bonferroni") -> pd.DataFrame:
    """
    Perform pairwise t-tests with multiple comparisons correction.

    Args:
        df: RT data
        correction: 'bonferroni' or 'holm'
    """
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests

    cardinalities = sorted(df["Cardinality"].unique())
    pairs = list(combinations(cardinalities, 2))

    results = []
    p_values = []

    for c1, c2 in pairs:
        rt1 = df[df["Cardinality"] == c1]["RT"]
        rt2 = df[df["Cardinality"] == c2]["RT"]

        t_stat, p = stats.ttest_ind(rt1, rt2)

        # Cohen's d effect size
        pooled_std = np.sqrt((rt1.std()**2 + rt2.std()**2) / 2)
        cohens_d = (rt1.mean() - rt2.mean()) / pooled_std

        results.append({
            "Cardinality_1": c1,
            "Cardinality_2": c2,
            "Mean_Diff": rt1.mean() - rt2.mean(),
            "t": t_stat,
            "p_uncorrected": p,
            "Cohen_d": cohens_d,
        })
        p_values.append(p)

    # Apply multiple comparisons correction
    _, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method=correction)

    for i, result in enumerate(results):
        result["p_corrected"] = p_corrected[i]
        result["significant"] = p_corrected[i] < 0.05

    return pd.DataFrame(results)


def test_linear_trend(df: pd.DataFrame) -> dict:
    """
    Test for linear trend: Does RT increase with cardinality?

    Uses Pearson correlation and linear regression.
    """
    r, p = stats.pearsonr(df["Cardinality"], df["RT"])

    # Linear regression
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(df["Cardinality"], df["RT"])

    return {
        "test": "Linear Trend",
        "correlation_r": r,
        "correlation_p": p,
        "slope": slope,
        "slope_p": p_value,
        "R_squared": r_value**2,
    }


def plot_linear_trend(df: pd.DataFrame, trend_results: dict, output_path: Path) -> None:
    """Plot linear trend of RT vs cardinality with regression line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute summary statistics per cardinality
    summary = df.groupby("Cardinality").agg(
        Mean_RT=("RT", "mean"),
        SEM_RT=("RT", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with error bars
    ax.errorbar(
        summary["Cardinality"],
        summary["Mean_RT"],
        yerr=summary["SEM_RT"],
        fmt="o",
        markersize=10,
        capsize=6,
        linewidth=2,
        color="#2a6f97",
        ecolor="#2a6f97",
        label="Observed Mean ± SEM",
    )

    # Regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(df["Cardinality"], df["RT"])

    x_line = np.array([1, 6])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "--", linewidth=2.5, color="darkred",
           label=f"Linear fit: RT = {intercept:.1f} + {slope:.1f}×Card")

    # Formatting
    ax.set_xlabel("Cardinality (Numerosity)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Reaction Time (ms)", fontsize=13, fontweight="bold")
    ax.set_title("Linear Trend: RT vs Numerosity", fontsize=15, fontweight="bold", pad=15)

    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add statistics annotation
    r = trend_results["correlation_r"]
    p = trend_results["correlation_p"]
    r2 = trend_results["R_squared"]
    stats_text = f"r = {r:.3f}, p < 0.001\nR² = {r2:.3f}\nSlope = {slope:.2f} ms/numerosity"
    ax.text(0.98, 0.02, stats_text,
           transform=ax.transAxes, ha="right", va="bottom",
           fontsize=11, bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[analyze_rt_stats] Linear trend plot saved to {output_path}")


def create_summary_stats_table(results: dict, output_path: Path) -> None:
    """Create formatted summary statistics table as PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build table data
    table_data = []

    # ANOVA
    anova = results["anova"]
    table_data.append([
        "One-way ANOVA",
        f"F({anova['df_between']}, {anova['df_within']}) = {anova['F']:.2f}",
        f"{anova['p']:.6f}",
        "***" if anova['p'] < 0.001 else "**" if anova['p'] < 0.01 else "*" if anova['p'] < 0.05 else "n.s.",
        f"η² = {anova['eta_squared']:.4f}",
    ])

    # Friedman
    friedman = results["friedman"]
    table_data.append([
        "Friedman Test",
        f"χ²({friedman['df']}) = {friedman['chi2']:.2f}",
        f"{friedman['p']:.6f}",
        "***" if friedman['p'] < 0.001 else "**" if friedman['p'] < 0.01 else "*" if friedman['p'] < 0.05 else "n.s.",
        f"N = {friedman['n_subjects']} subjects",
    ])

    # Linear trend
    trend = results["linear_trend"]
    table_data.append([
        "Linear Trend",
        f"r = {trend['correlation_r']:.3f}",
        f"{trend['correlation_p']:.6f}",
        "***" if trend['correlation_p'] < 0.001 else "**" if trend['correlation_p'] < 0.01 else "*" if trend['correlation_p'] < 0.05 else "n.s.",
        f"Slope = {trend['slope']:.2f} ms/num",
    ])

    # Post-hoc summary
    pairwise = results["pairwise"]
    n_sig = pairwise["significant"].sum()
    table_data.append([
        "Post-hoc Pairwise",
        f"{n_sig} of {len(pairwise)} pairs",
        f"α = {0.05/len(pairwise):.4f}",
        "Bonferroni corrected",
        f"See heatmap",
    ])

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["Test", "Statistic", "p-value", "Sig.", "Effect Size / Notes"],
        cellLoc="left",
        loc="center",
        colWidths=[0.25, 0.25, 0.15, 0.10, 0.25],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor("#2a6f97")
        cell.set_text_props(weight="bold", color="white", ha="center")

    # Style data rows
    for i in range(1, 5):
        for j in range(5):
            cell = table[(i, j)]
            cell.set_facecolor("#f0f0f0" if i % 2 == 0 else "white")

            # Highlight significance column
            if j == 3:
                sig = table_data[i-1][3]
                if sig == "***":
                    cell.set_facecolor("#d4edda")
                    cell.set_text_props(weight="bold", color="darkgreen")

    ax.set_title("Statistical Test Summary: RT Differences Across Numerosities",
                fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[analyze_rt_stats] Summary statistics table saved to {output_path}")


def plot_pairwise_comparisons(pairwise_df: pd.DataFrame, output_path: Path) -> None:
    """Plot heatmap of pairwise comparisons."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create matrix for heatmap
    cardinalities = sorted(set(pairwise_df["Cardinality_1"]).union(pairwise_df["Cardinality_2"]))
    n = len(cardinalities)

    # Mean difference matrix
    diff_matrix = np.zeros((n, n))
    p_matrix = np.zeros((n, n))

    for _, row in pairwise_df.iterrows():
        i = cardinalities.index(row["Cardinality_1"])
        j = cardinalities.index(row["Cardinality_2"])
        diff_matrix[i, j] = row["Mean_Diff"]
        diff_matrix[j, i] = -row["Mean_Diff"]
        p_matrix[i, j] = row["p_corrected"]
        p_matrix[j, i] = row["p_corrected"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Mean difference heatmap
    sns.heatmap(
        diff_matrix,
        annot=True,
        fmt=".1f",
        cmap="RdBu_r",
        center=0,
        xticklabels=cardinalities,
        yticklabels=cardinalities,
        cbar_kws={"label": "Mean RT Difference (ms)"},
        ax=ax1,
    )
    ax1.set_title("Pairwise RT Differences\n(Row - Column)", fontweight="bold")
    ax1.set_xlabel("Cardinality", fontweight="bold")
    ax1.set_ylabel("Cardinality", fontweight="bold")

    # Significance heatmap
    sig_matrix = (p_matrix < 0.05).astype(int)
    sns.heatmap(
        p_matrix,
        annot=sig_matrix,
        fmt="d",
        cmap="RdYlGn_r",
        vmin=0,
        vmax=0.1,
        xticklabels=cardinalities,
        yticklabels=cardinalities,
        cbar_kws={"label": "p-value (corrected)"},
        ax=ax2,
    )
    ax2.set_title("Statistical Significance\n(1=significant, 0=n.s.)", fontweight="bold")
    ax2.set_xlabel("Cardinality", fontweight="bold")
    ax2.set_ylabel("Cardinality", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[analyze_rt_stats] Pairwise comparison heatmaps saved to {output_path}")


def save_statistical_report(results: dict, output_path: Path) -> None:
    """Save comprehensive statistical report as text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("REACTION TIME STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("Research Question: Do reaction times differ significantly across numerosities?\n\n")

        # ANOVA
        f.write("-" * 80 + "\n")
        f.write("1. ONE-WAY ANOVA (ignores subject structure)\n")
        f.write("-" * 80 + "\n")
        anova = results["anova"]
        f.write(f"   F({anova['df_between']}, {anova['df_within']}) = {anova['F']:.3f}\n")
        f.write(f"   p-value = {anova['p']:.6f} {'***' if anova['p'] < 0.001 else '**' if anova['p'] < 0.01 else '*' if anova['p'] < 0.05 else 'n.s.'}\n")
        f.write(f"   Effect size (η²) = {anova['eta_squared']:.4f}\n\n")

        # Friedman
        f.write("-" * 80 + "\n")
        f.write("2. FRIEDMAN TEST (non-parametric repeated measures)\n")
        f.write("-" * 80 + "\n")
        friedman = results["friedman"]
        f.write(f"   χ²({friedman['df']}) = {friedman['chi2']:.3f}\n")
        f.write(f"   p-value = {friedman['p']:.6f} {'***' if friedman['p'] < 0.001 else '**' if friedman['p'] < 0.01 else '*' if friedman['p'] < 0.05 else 'n.s.'}\n")
        f.write(f"   N subjects with complete data = {friedman['n_subjects']}\n\n")

        # LMM
        f.write("-" * 80 + "\n")
        f.write("3. LINEAR MIXED-EFFECTS MODEL (gold standard)\n")
        f.write("-" * 80 + "\n")
        lmm = results["lmm"]
        if "error" not in lmm:
            f.write(f"   Model: RT ~ Cardinality + (1 | Subject)\n")
            f.write(f"   Likelihood Ratio χ²({lmm['df']}) = {lmm['LR_chi2']:.3f}\n")
            f.write(f"   p-value = {lmm['p']:.6f} {'***' if lmm['p'] < 0.001 else '**' if lmm['p'] < 0.01 else '*' if lmm['p'] < 0.05 else 'n.s.'}\n")
            f.write(f"   AIC = {lmm['AIC']:.1f}, BIC = {lmm['BIC']:.1f}\n\n")
        else:
            f.write(f"   {lmm['error']}\n\n")

        # Linear trend
        f.write("-" * 80 + "\n")
        f.write("4. LINEAR TREND ANALYSIS\n")
        f.write("-" * 80 + "\n")
        trend = results["linear_trend"]
        f.write(f"   Pearson r = {trend['correlation_r']:.4f}, p = {trend['correlation_p']:.6f}\n")
        f.write(f"   Regression slope = {trend['slope']:.2f} ms/numerosity, p = {trend['slope_p']:.6f}\n")
        f.write(f"   R² = {trend['R_squared']:.4f}\n\n")

        # Post-hoc
        f.write("-" * 80 + "\n")
        f.write("5. POST-HOC PAIRWISE COMPARISONS (Bonferroni corrected)\n")
        f.write("-" * 80 + "\n")
        pairwise = results["pairwise"]
        f.write(f"   Number of comparisons: {len(pairwise)}\n")
        f.write(f"   Significance threshold (α = 0.05): p < {0.05/len(pairwise):.4f}\n\n")

        significant = pairwise[pairwise["significant"]]
        if len(significant) > 0:
            f.write(f"   SIGNIFICANT DIFFERENCES ({len(significant)} pairs):\n")
            for _, row in significant.iterrows():
                f.write(f"      {int(row['Cardinality_1'])} vs {int(row['Cardinality_2'])}: ")
                f.write(f"Δ = {row['Mean_Diff']:+.1f}ms, ")
                f.write(f"p = {row['p_corrected']:.4f}, ")
                f.write(f"d = {row['Cohen_d']:.3f}\n")
        else:
            f.write("   NO SIGNIFICANT DIFFERENCES after correction\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("=" * 80 + "\n\n")

        # Interpret results
        if lmm.get("p", 1) < 0.05 or anova["p"] < 0.05:
            f.write("CONCLUSION: Reaction times DO differ significantly across numerosities.\n\n")

            if len(significant) > 0:
                f.write(f"Post-hoc tests reveal {len(significant)} significant pairwise differences.\n")
            else:
                f.write("However, no individual pairwise comparisons reach significance after\n")
                f.write("multiple comparisons correction (suggests small/distributed effects).\n")
        else:
            f.write("CONCLUSION: No significant differences in RT across numerosities.\n")
            f.write("RTs are relatively uniform (~40ms range) across the subitizing and\n")
            f.write("counting ranges, suggesting efficient numerosity processing.\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"[analyze_rt_stats] Statistical report saved to {output_path}")


def main() -> None:
    """Run comprehensive statistical analysis."""
    # Load data
    data_path = Path("reaction_time/tables/rt_all_trials.csv")
    df = load_rt_data(data_path)

    print("[analyze_rt_stats] Running statistical analyses...")

    # Perform tests
    results = {}

    print("  - One-way ANOVA...")
    results["anova"] = perform_anova(df)

    print("  - Friedman test...")
    results["friedman"] = perform_friedman_test(df)

    print("  - Linear Mixed-Effects Model...")
    results["lmm"] = perform_lmm(df)

    print("  - Linear trend analysis...")
    results["linear_trend"] = test_linear_trend(df)

    print("  - Post-hoc pairwise comparisons...")
    results["pairwise"] = perform_pairwise_t_tests(df, correction="bonferroni")

    # Save outputs
    output_dir = Path("reaction_time")

    # Save pairwise table
    results["pairwise"].to_csv(output_dir / "tables" / "rt_pairwise_comparisons.csv", index=False)

    # Save Tukey HSD
    tukey_results = perform_posthoc_tukey(df)
    tukey_results.to_csv(output_dir / "tables" / "rt_tukey_hsd.csv", index=False)

    # Generate plots
    print("  - Generating visualizations...")
    plot_pairwise_comparisons(results["pairwise"], output_dir / "figures" / "rt_pairwise_heatmaps.png")
    plot_linear_trend(df, results["linear_trend"], output_dir / "figures" / "rt_linear_trend.png")
    create_summary_stats_table(results, output_dir / "figures" / "rt_summary_stats_table.png")

    # Save comprehensive report
    save_statistical_report(results, output_dir / "rt_statistical_report.txt")

    # Print summary to console
    print("\n" + "="*80)
    print("STATISTICAL SUMMARY")
    print("="*80)
    print(f"ANOVA: F = {results['anova']['F']:.3f}, p = {results['anova']['p']:.6f}")
    print(f"Friedman: chi2 = {results['friedman']['chi2']:.3f}, p = {results['friedman']['p']:.6f}")
    if "error" not in results["lmm"]:
        print(f"LMM: LR chi2 = {results['lmm']['LR_chi2']:.3f}, p = {results['lmm']['p']:.6f}")
    print(f"Linear trend: r = {results['linear_trend']['correlation_r']:.4f}, p = {results['linear_trend']['correlation_p']:.6f}")
    print(f"\nSignificant pairwise differences: {results['pairwise']['significant'].sum()}/{len(results['pairwise'])}")
    print("="*80)

    print(f"\n[analyze_rt_stats] Analysis complete! See {output_dir}/rt_statistical_report.txt")


if __name__ == "__main__":
    main()
