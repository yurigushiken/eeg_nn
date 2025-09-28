#!/usr/bin/env python
from __future__ import annotations
import argparse
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple
import math
import csv
import numpy as np
import sys

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Silence rpy2 CFFI mode message on Windows by forcing ABI before any rpy2 import
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

# add reporting for HTML refresh
try:
    from utils.reporting import create_consolidated_reports
except Exception:
    create_consolidated_reports = None

def fig_to_png_bytes(fig) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


def _read_outer_eval_metrics(run_dir: Path) -> List[Dict]:
    fp = run_dir / "outer_eval_metrics.csv"
    rows: List[Dict] = []
    if not fp.exists():
        return rows
    with fp.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("outer_fold") == "OVERALL":
                continue
            try:
                rows.append({
                    "outer_fold": int(row["outer_fold"]),
                    "acc": float(row["acc"]),
                    "macro_f1": float(row["macro_f1"]),
                    "n_test_trials": int(row.get("n_test_trials", 0) or 0),
                })
            except Exception:
                pass
    return rows


def _read_test_predictions(run_dir: Path) -> List[Dict]:
    fp = run_dir / "test_predictions.csv"
    rows: List[Dict] = []
    if not fp.exists():
        return rows
    with fp.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "subject_id": int(row["subject_id"]),
                    "correct": int(row["correct"]),
                })
            except Exception:
                pass
    return rows


def _binom_test_greater(k: int, n: int, p: float) -> float:
    # Exact binomial test, right-tailed; fallback when SciPy is unavailable
    # Sum_{x=k..n} C(n,x) p^x (1-p)^(n-x)
    if n <= 0:
        return 1.0
    # Handle edge cases
    k = max(0, min(k, n))
    total = 0.0
    for x in range(k, n + 1):
        total += math.comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
    return float(min(1.0, max(0.0, total)))


def _wilson_ci(k: int, n: int, alpha: float) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # approx for 95% (use alpha arg if needed later)
    phat = k / n
    denom = 1 + z**2/n
    center = phat + z*z/(2*n)
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return float(lower*100.0), float(upper*100.0)


def _ci95_mean(vals: List[float]) -> Tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    m = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = 1.959963984540054 * (sd / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return m, m - half, m + half


# (removed legacy _bh_fdr helper; using statsmodels' multipletests instead)


def compute_group_stats(run_dir: Path, alpha: float, permute_parent: bool) -> Dict:
    folds = _read_outer_eval_metrics(run_dir)
    acc_vals = [r["acc"] for r in folds]
    f1_vals = [r["macro_f1"] for r in folds]
    mean_acc, lo_acc, hi_acc = _ci95_mean(acc_vals)
    mean_f1, lo_f1, hi_f1 = _ci95_mean(f1_vals)

    # Try to read permutation results from parent
    perm_summary = None
    if permute_parent:
        cand = run_dir.parent / f"{run_dir.name}_perm_summary.json"
        if cand.exists():
            try:
                perm_summary = json.loads(cand.read_text())
            except Exception:
                perm_summary = None

    return {
        "mean_acc": mean_acc,
        "ci95_acc": [lo_acc, hi_acc],
        "mean_macro_f1": mean_f1,
        "ci95_macro_f1": [lo_f1, hi_f1],
        "perm_p_acc": (perm_summary or {}).get("p_value_acc"),
        "perm_p_macro_f1": (perm_summary or {}).get("p_value_macro_f1"),
        "bookkeeping": {
            "n_folds": len(folds),
        },
    }


def compute_per_subject_significance(run_dir: Path, chance_rate: float, alpha: float, multitest: str) -> Tuple[List[Dict], Dict]:
    rows = _read_test_predictions(run_dir)
    if not rows:
        return [], {"num_subjects": 0, "num_above": 0, "proportion_above": 0.0}
    # Aggregate by subject
    by_subj: Dict[int, Tuple[int,int]] = {}  # subj -> (k_correct, n)
    for r in rows:
        sid = int(r["subject_id"]) 
        correct = int(r["correct"]) 
        k, n = by_subj.get(sid, (0, 0))
        by_subj[sid] = (k + (1 if correct else 0), n + 1)

    out_rows: List[Dict] = []
    pvals: List[float] = []
    for sid, (k, n) in sorted(by_subj.items()):
        acc = 100.0 * k / n if n > 0 else 0.0
        p = _binom_test_greater(k, n, chance_rate)
        pvals.append(p)
        out_rows.append({
            "subject_id": sid,
            "n_test_trials": n,
            "acc": acc,
            "p_binomial": p,
        })

    # Adjust p-values if requested (use vetted library implementation)
    if multitest.lower() == "fdr":
        try:
            from statsmodels.stats.multitest import multipletests
            reject, qvals, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
        except Exception:
            # Fallback to unadjusted if library not available
            reject = [pv < alpha for pv in pvals]
            qvals = pvals
    else:
        reject = [pv < alpha for pv in pvals]
        qvals = pvals
    num_above = 0
    for i, row in enumerate(out_rows):
        q = qvals[i]
        row["p_adj"] = q
        # If FDR was applied, prefer reject mask when available
        row["above_chance"] = bool(reject[i]) if 'reject' in locals() else bool(q < alpha)
        if row["above_chance"]:
            num_above += 1

    footer = {
        "num_subjects": len(out_rows),
        "num_above": num_above,
        "proportion_above": (num_above / len(out_rows)) if out_rows else 0.0,
    }
    return out_rows, footer


def run_glmm(run_dir: Path, enable: bool, chance_rate: float) -> Dict:
    """Fit a logistic GLMM via R lme4::glmer using rpy2 (no fallbacks).

    Model tests above-chance performance using an offset at logit(chance):
        correct ~ 1 + offset(offset_term) + (1 | subject_id)
    where offset_term = logit(chance_rate).
    """
    if not enable:
        return {"status": "disabled"}

    fp = run_dir / "test_predictions.csv"
    if not fp.exists():
        return {"status": "no_test_predictions"}

    import pandas as pd
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        return {"status": "read_error", "message": str(e)}

        if "correct" not in df.columns or "subject_id" not in df.columns:
            return {"status": "columns_missing"}

    try:
        import math
        from rpy2 import robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr
        from rpy2.robjects.conversion import localconverter
    except Exception as e:
        return {"status": "rpy2_not_available", "message": str(e)}

    # Prepare data and offset
    try:
        df = df[["correct", "subject_id"]].dropna()
        df["correct"] = df["correct"].astype(int)
        p0 = float(chance_rate if (chance_rate is not None and 0.0 < chance_rate < 1.0) else 0.5)
        # constant offset vector in the data frame to satisfy R's offset()
        logit_p0 = math.log(p0 / (1.0 - p0))
        df["offset_term"] = float(logit_p0)
    except Exception as e:
        return {"status": "prep_error", "message": str(e)}

    try:
        # Convert pandas -> R using local converter (no deprecated global activation)
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df)

        lme4 = importr("lme4")
        base = importr("base")
        stats = importr("stats")

        ro.globalenv["d"] = r_df
        ro.r("""
            d$correct <- as.integer(d$correct)
            d$subject_id <- as.factor(d$subject_id)
            fit <- lme4::glmer(correct ~ 1 + offset(offset_term) + (1 | subject_id),
                                data=d, family=stats::binomial())
        """)
        fit = ro.globalenv["fit"]

        # Extract coefficients table using R expression to avoid generic dispatch issues
        coef_df_r = ro.r('as.data.frame(coef(summary(fit)))')
        with localconverter(ro.default_converter + pandas2ri.converter):
            coef_df = ro.conversion.rpy2py(coef_df_r)
        # Ensure row names
        coef_df.index = [str(x) for x in coef_df.index]
        row = coef_df.loc['(Intercept)'] if '(Intercept)' in coef_df.index else coef_df.iloc[0]
        est = float(row.get('Estimate', row.iloc[0]))
        se = float(row.get('Std. Error', float('nan')))
        zval = float(row.get('z value', float('nan')))
        pval = float(row.get('Pr(>|z|)', float('nan')))

        # Probability at intercept relative to chance
        p_at_intercept = float(1.0 / (1.0 + math.exp(-(logit_p0 + est))))

        # Random effect variance for subject
        vc_df_r = ro.r("as.data.frame(lme4::VarCorr(fit))")
        with localconverter(ro.default_converter + pandas2ri.converter):
            vc_df = ro.conversion.rpy2py(vc_df_r)
        re_var = None
        try:
            re_row = vc_df.loc[vc_df['grp'] == 'subject_id']
            if not re_row.empty:
                re_var = float(re_row.iloc[0]['vcov']) if 'vcov' in re_row.columns else None
        except Exception:
            re_var = None

        # Extract subject random intercept BLUPs (+ conditional SEs)
        re_r = ro.r('''
          re <- lme4::ranef(fit, condVar = TRUE)$subject_id
          pv <- attr(re, "postVar")
          # pv should be a 3D array with dims [coef, coef, subjects] when only intercepts exist => [1,1,N]
          se <- tryCatch(sqrt(pv[1,1,]), error=function(e) {
            # Fallback: if not 3D, coerce via as.vector
            sqrt(as.vector(pv))
          })
          data.frame(subject = rownames(re),
                     blup = as.numeric(re[,1]),
                     se = as.numeric(se),
                     row.names = NULL)
        ''')
        with localconverter(ro.default_converter + pandas2ri.converter):
            re_df = ro.conversion.rpy2py(re_r)
        ranef_payload = []
        try:
            for subj, bl, sv in zip(re_df["subject"], re_df["blup"], re_df["se"]):
                ranef_payload.append({"subject": str(subj), "blup": float(bl), "se": float(sv)})
        except Exception:
            ranef_payload = []

        return {
            "status": "ok_glmm",
            "engine": "R lme4::glmer via rpy2",
            "formula": "correct ~ 1 + offset(qlogis(chance)) + (1 | subject_id)",
            "chance_rate": p0,
            "intercept_delta_logit_from_chance": est,
            "intercept_se": se,
            "intercept_z": zval,
            "intercept_pvalue": pval,
            "probability_at_intercept": p_at_intercept,
            "random_effects_variance_subject": re_var,
            "ranef_subject": ranef_payload,
        }
    except Exception as e:
        return {"status": "fit_error", "message": str(e)}


def write_json(p: Path, obj: Dict):
    p.write_text(json.dumps(obj, indent=2))


def write_csv(p: Path, rows: List[Dict], fieldnames: List[str]):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Post-hoc stats for a completed run directory.")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--chance-rate", type=float, default=None, help="If unset, uses 1/num_classes from summary")
    ap.add_argument("--multitest", choices=["none", "fdr"], default="fdr")
    ap.add_argument("--glmm", action="store_true", help="Attempt GLMM (or cluster-robust logit fallback)")
    ap.add_argument("--forest", action="store_true", help="Write per_subject_forest.png if matplotlib available")
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        sys.exit(f"Run directory not found: {run_dir}")
    # All outputs now go under <run_dir>/stats
    stats_dir = run_dir / "stats"
    stats_dir.mkdir(exist_ok=True)

    # Load summary to get class_names and seeds if needed
    try:
        summary_path = next(run_dir.glob("summary_*.json"))
        summary = json.loads(summary_path.read_text())
    except Exception:
        summary = {}
    class_names = summary.get("hyper", {}).get("class_names") or summary.get("class_names")
    if args.chance_rate is not None:
        chance_rate = float(args.chance_rate)
    else:
        if isinstance(class_names, list) and len(class_names) > 0:
            chance_rate = 1.0 / float(len(class_names))
        else:
            # Try to infer from task CONDITIONS in summary or by importing the task module
            n_classes = None
            try:
                # summary written by engines/eeg.py may include num_classes
                n_classes = int(summary.get("num_classes")) if summary.get("num_classes") else None
            except Exception:
                n_classes = None
            if not n_classes:
                try:
                    task_name = summary.get("hyper", {}).get("task") or summary.get("task")
                    if isinstance(task_name, str) and len(task_name) > 0:
                        import importlib
                        task_mod = importlib.import_module(f"tasks.{task_name}")
                        conds = getattr(task_mod, "CONDITIONS", None)
                        if isinstance(conds, (list, tuple)) and len(conds) > 0:
                            n_classes = len(conds)
                except Exception:
                    n_classes = None
            chance_rate = 1.0 / float(n_classes) if n_classes and n_classes > 0 else 0.5

    # Group-level stats
    group = compute_group_stats(run_dir, args.alpha, permute_parent=True)
    group["bookkeeping"].update({
        "chance_rate": chance_rate,
    })
    write_json(stats_dir / "group_stats.json", group)

    # Subject-level stats
    per_subj_rows, footer = compute_per_subject_significance(run_dir, chance_rate, args.alpha, args.multitest)
    write_csv(stats_dir / "per_subject_significance.csv", per_subj_rows, [
        "subject_id", "n_test_trials", "acc", "p_binomial", "p_adj", "above_chance",
    ])
    write_json(stats_dir / "per_subject_summary.json", footer)

    # Optional forest plot (to stats/)
    if args.forest:
        if plt is None:
            print("[posthoc] matplotlib not available; skipping forest plot.")
        elif not per_subj_rows:
            print("[posthoc] No per-subject rows; skipping forest plot.")
        else:
            try:
                stats_dir.mkdir(exist_ok=True)
                ids = [int(r["subject_id"]) for r in per_subj_rows]
                accs = [float(r["acc"]) for r in per_subj_rows]
                ns = [int(r["n_test_trials"]) for r in per_subj_rows]
                # Wilson CI per subject
                ci = [_wilson_ci(int(round(a/100.0*n)), n, 0.05) for a, n in zip(accs, ns)]
                lowers = [c[0] for c in ci]; uppers = [c[1] for c in ci]
                y = np.arange(len(ids))
                fig, ax = plt.subplots(figsize=(6, max(4, 0.3*len(ids))))
                ax.errorbar(accs, y, xerr=[np.array(accs)-np.array(lowers), np.array(uppers)-np.array(accs)], fmt='o', color='black')
                ax.axvline(100.0*chance_rate, color='red', linestyle='--', label='Chance')
                ax.set_xlabel('Accuracy (%)')
                ax.set_yticks(y)
                ax.set_yticklabels([str(i) for i in ids])
                ax.invert_yaxis()
                ax.legend(loc='lower right')
                plt.tight_layout()
                out_png = stats_dir / "per_subject_forest.png"
                out_png.write_bytes(fig_to_png_bytes(fig))
                plt.close(fig)
                print(f"[posthoc] Wrote forest plot: {out_png}")
            except Exception as e:
                print(f"[posthoc] Forest plot failed: {e}")

    # Optional permutation densities (if perm results CSV exists in parent)
    try:
        perm_csv = run_dir.parent / f"{run_dir.name}_perm_test_results.csv"
        if perm_csv.exists() and plt is not None:
            # compute per-permutation overall means from CSV
            import pandas as pd
            df = pd.read_csv(perm_csv)
            if {"perm_id", "acc", "macro_f1"}.issubset(df.columns):
                stats_dir.mkdir(exist_ok=True)
                acc_by_perm = df.groupby("perm_id")["acc"].mean().values
                f1_by_perm = df.groupby("perm_id")["macro_f1"].mean().values
                # Observed means and permutation p-values
                obs_acc = float(group.get("mean_acc", 0.0))
                obs_f1 = float(group.get("mean_macro_f1", 0.0))
                p_perm_acc = float((np.sum(acc_by_perm >= obs_acc) + 1) / (acc_by_perm.size + 1)) if acc_by_perm.size else 1.0
                p_perm_f1 = float((np.sum(f1_by_perm >= obs_f1) + 1) / (f1_by_perm.size + 1)) if f1_by_perm.size else 1.0
                # Plot density-style histograms with observed vertical line
                fig, ax = plt.subplots(figsize=(5,3))
                ax.hist(acc_by_perm, bins=24, color="#4C78A8", alpha=0.6, density=True)
                ax.axvline(obs_acc, color='red', linestyle='--', linewidth=2)
                ax.set_title(f"Null density of accuracy (perm); p={p_perm_acc:.3f}")
                ax.set_xlabel("Accuracy (%)")
                ax.set_ylabel("Density")
                plt.tight_layout()
                (stats_dir / "perm_density_acc.png").write_bytes(fig_to_png_bytes(fig))
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(5,3))
                ax.hist(f1_by_perm, bins=24, color="#F58518", alpha=0.6, density=True)
                ax.axvline(obs_f1, color='red', linestyle='--', linewidth=2)
                ax.set_title(f"Null density of macro-F1 (perm); p={p_perm_f1:.3f}")
                ax.set_xlabel("Macro-F1 (%)")
                ax.set_ylabel("Density")
                plt.tight_layout()
                (stats_dir / "perm_density_macro_f1.png").write_bytes(fig_to_png_bytes(fig))
                plt.close(fig)
    except Exception:
        pass

    # GLMM (true GLMM via R lme4)
    glmm = run_glmm(run_dir, args.glmm, chance_rate)
    write_json(stats_dir / "glmm_summary.json", glmm)

    # Additional visuals linked to GLMM and p-values
    try:
        if plt is not None:
            # 1) QQ plot of per-subject p-values with BH threshold
            if per_subj_rows:
                pvals_arr = np.array([float(r["p_binomial"]) for r in per_subj_rows], dtype=float)
                m = pvals_arr.size
                if m > 0:
                    p_sorted = np.sort(pvals_arr)
                    exp = (np.arange(1, m+1) - 0.5) / m
                    alpha = float(args.alpha)
                    # BH cutoff line y = (k/m)*alpha at the largest k where p_(k) <= (k/m)*alpha
                    thresh_idx = np.where(p_sorted <= (np.arange(1, m+1)/m)*alpha)[0]
                    fig, ax = plt.subplots(figsize=(4.8, 4.8))
                    ax.plot(exp, p_sorted, 'o', color="#1f77b4", markersize=4)
                    ax.plot([0,1],[0,1], '--', color='gray')
                    if thresh_idx.size:
                        k = int(thresh_idx.max()) + 1
                        yline = (k/m)*alpha
                        ax.axhline(yline, color='red', linestyle='--')
                        ax.text(0.05, min(0.95, yline+0.03), f"BH cutoff (k={k})", color='red')
                    ax.set_xlabel('Expected p (Uniform)')
                    ax.set_ylabel('Observed p (Binomial)')
                    ax.set_title('Per-subject p-value QQ')
                    plt.tight_layout()
                    (stats_dir / "qq_pvalues_fdr.png").write_bytes(fig_to_png_bytes(fig))
                    plt.close(fig)

            # 2) GLMM visuals if model succeeded
            if glmm.get("status") == "ok_glmm":
                # Intercept effect panel (log-odds with CI, prob scale on right)
                est = float(glmm.get("intercept_delta_logit_from_chance", 0.0))
                se = float(glmm.get("intercept_se", float('nan')))
                p0 = float(glmm.get("chance_rate", chance_rate))
                if se == se:  # not NaN
                    lo = est - 1.959963984540054*se
                    hi = est + 1.959963984540054*se
                    logit_p0 = math.log(p0/(1-p0))
                    fig, ax = plt.subplots(figsize=(6.0, 3.2))
                    ax.errorbar([0], [est], yerr=[[est-lo],[hi-est]], fmt='o', color='#333')
                    ax.axhline(0.0, color='gray', linestyle=':')
                    ax.set_xticks([0])
                    ax.set_xticklabels(["Intercept vs chance (log-odds)"])
                    ax.set_ylabel("Estimate (log-odds)")
                    # Secondary y-axis showing probability at intercept
                    def logit_to_prob(x):
                        x = np.asarray(x, dtype=float)
                        return 1.0/(1.0+np.exp(-(logit_p0 + x)))
                    def prob_to_logit(p):
                        p = np.asarray(p, dtype=float)
                        return np.log(p/(1-p)) - logit_p0
                    ax.secondary_yaxis('right', functions=(logit_to_prob, prob_to_logit)).set_ylabel('Probability at intercept')
                    plt.tight_layout()
                    (stats_dir / "glmm_intercept_effect.png").write_bytes(fig_to_png_bytes(fig))
                    plt.close(fig)

                # Caterpillar plot of subject BLUPs on probability scale (if available)
                ranef = glmm.get("ranef_subject", [])
                if ranef:
                    import pandas as pd
                    re_df = pd.DataFrame(ranef)
                    if {"subject","blup"}.issubset(re_df.columns):
                        # Use SE when provided; otherwise draw points only
                        bl = re_df["blup"].astype(float)
                        se_arr = re_df["se"].astype(float) if "se" in re_df.columns else None
                        logit_p0 = math.log(p0/(1-p0))
                        prob = 1.0 / (1.0 + np.exp(-(logit_p0 + est + bl)))
                        if se_arr is not None:
                            prob_lo = 1.0 / (1.0 + np.exp(-(logit_p0 + est + (bl - 1.959963984540054*se_arr))))
                            prob_hi = 1.0 / (1.0 + np.exp(-(logit_p0 + est + (bl + 1.959963984540054*se_arr))))
                        order = np.argsort(prob)
                        y = np.arange(len(order))
                        fig, ax = plt.subplots(figsize=(6.4, max(4.0, 0.35*len(order))))
                        if se_arr is not None:
                            ax.hlines(y, prob_lo.values[order], prob_hi.values[order], color="#555")
                        ax.plot(prob.values[order], y, 'o', color="#1f77b4")
                        ax.axvline(p0, color='red', linestyle='--', label='Chance')
                        ax.set_xlabel('Probability')
                        ax.set_yticks(y)
                        ax.set_yticklabels(re_df["subject"].astype(str).values[order].tolist())
                        ax.invert_yaxis()
                        ax.legend(loc='lower right')
                        plt.tight_layout()
                        (stats_dir / "glmm_caterpillar.png").write_bytes(fig_to_png_bytes(fig))
                        plt.close(fig)
    except Exception as e:
        print(f"[posthoc] Additional visuals failed: {e}")

    print("[posthoc] Wrote stats outputs to stats/ (group_stats.json, per_subject_significance.csv, per_subject_summary.json, glmm_summary.json)")

    # Refresh consolidated report to include stats section if possible
    try:
        if create_consolidated_reports is not None:
            # Load summary for banner/context
            summary_path = next(run_dir.glob("summary_*.json"))
            summary = json.loads(summary_path.read_text())
            task = summary.get("hyper", {}).get("task") or ""
            engine = summary.get("hyper", {}).get("engine") or "eeg"
            if task:
                create_consolidated_reports(run_dir, summary, task, engine)
                print("[posthoc] Consolidated report refreshed with statistics.")
    except Exception as e:
        print(f"[posthoc] Could not refresh consolidated report: {e}")


if __name__ == "__main__":
    main()


