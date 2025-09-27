#!/usr/bin/env python
from __future__ import annotations
import argparse
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


def _bh_fdr(pvals: List[float], alpha: float) -> List[float]:
    # Benjamini-Hochberg adjusted p-values (q-values)
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.empty(m, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order, start=1):
        pv = pvals[idx]
        q = pv * m / rank
        if rank == 1:
            ranked[idx] = q
            prev = q
        else:
            ranked[idx] = min(q, prev)
            prev = ranked[idx]
    # Ensure in [0,1]
    ranked = np.clip(ranked, 0.0, 1.0)
    return ranked.tolist()


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

    # Adjust p-values if requested
    if multitest.lower() == "fdr":
        qvals = _bh_fdr(pvals, alpha)
    else:
        qvals = pvals
    num_above = 0
    for i, row in enumerate(out_rows):
        q = qvals[i]
        row["p_adj"] = q
        row["above_chance"] = bool(q < alpha)
        if row["above_chance"]:
            num_above += 1

    footer = {
        "num_subjects": len(out_rows),
        "num_above": num_above,
        "proportion_above": (num_above / len(out_rows)) if out_rows else 0.0,
    }
    return out_rows, footer


def maybe_glmm(run_dir: Path, enable: bool) -> Dict:
    if not enable:
        return {"status": "disabled"}
    # Fallback: cluster-robust logistic regression at subject level if GLMM unavailable
    try:
        import pandas as pd
        import statsmodels.api as sm
        from statsmodels.discrete.discrete_model import Logit
    except Exception:
        return {"status": "statsmodels_not_available"}
    fp = run_dir / "test_predictions.csv"
    if not fp.exists():
        return {"status": "no_test_predictions"}
    import pandas as pd
    df = pd.read_csv(fp)
    if "correct" not in df.columns or "subject_id" not in df.columns:
        return {"status": "columns_missing"}
    try:
        df = df[["correct", "subject_id"]].dropna()
        df["intercept"] = 1.0
        model = Logit(df["correct"], df[["intercept"]])
        res = model.fit(disp=False, method="newton")
        # Cluster-robust SEs by subject (approximate random effects via clusters)
        robust = res.get_robustcov_results(cov_type="cluster", groups=df["subject_id"])  
        coef = float(robust.params[0])
        se = float(robust.bse[0])
        z = coef / se if se > 0 else 0.0
        from math import erf, sqrt
        pval = float(2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2)))))
        # Convert intercept log-odds to accuracy at baseline
        acc_at_intercept = float(100.0 * (1.0 / (1.0 + math.exp(-coef))))
        ci_low = float(100.0 * (1.0 / (1.0 + math.exp(-(coef - 1.959963984540054 * se)))))
        ci_hi = float(100.0 * (1.0 / (1.0 + math.exp(-(coef + 1.959963984540054 * se)))))
        # Approximate random-effects variance via cluster variance proxy if available
        re_var = None
        try:
            # statsmodels doesn't directly give RE var in this setup; leave None
            re_var = None
        except Exception:
            re_var = None
        return {
            "status": "ok_cluster_robust",
            "intercept_log_odds": coef,
            "intercept_pvalue": pval,
            "acc_at_intercept_pct": acc_at_intercept,
            "ci95_acc_at_intercept_pct": [ci_low, ci_hi],
            "random_effects_variance_subject": re_var,
        }
    except Exception as e:
        return {"status": f"failed: {e}"}


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
            chance_rate = 0.5  # fallback

    # Group-level stats
    group = compute_group_stats(run_dir, args.alpha, permute_parent=True)
    group["bookkeeping"].update({
        "chance_rate": chance_rate,
    })
    write_json(run_dir / "group_stats.json", group)

    # Subject-level stats
    per_subj_rows, footer = compute_per_subject_significance(run_dir, chance_rate, args.alpha, args.multitest)
    write_csv(run_dir / "per_subject_significance.csv", per_subj_rows, [
        "subject_id", "n_test_trials", "acc", "p_binomial", "p_adj", "above_chance",
    ])
    write_json(run_dir / "per_subject_summary.json", footer)

    # Optional forest plot
    if args.forest and plt is not None and per_subj_rows:
        try:
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
            (run_dir / "per_subject_forest.png").write_bytes(fig_to_png_bytes(fig))
            plt.close(fig)
        except Exception:
            pass

    # Optional GLMM (or cluster-robust logit)
    glmm = maybe_glmm(run_dir, args.glmm)
    write_json(run_dir / "glmm_summary.json", glmm)

    print("[posthoc] Wrote group_stats.json, per_subject_significance.csv, per_subject_summary.json, glmm_summary.json")


if __name__ == "__main__":
    main()

def fig_to_png_bytes(fig) -> bytes:
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


