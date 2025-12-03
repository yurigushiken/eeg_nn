# Statistics Guide: Post-Hoc Analysis

Guide to post-hoc statistical analysis for quantifying population-level efficacy and subject-level reliability.

## Overview

The post-hoc statistics module provides:
1. **Group-level efficacy:** Mean performance + 95% CI, permutation-based p-values
2. **Subject-level reliability:** Per-subject binomial tests (FDR-corrected)
3. **GLMM analysis:** Population fixed effects using R's lme4 (optional)
4. **Forest plots:** Per-subject accuracy with Wilson confidence intervals
5. **Caterpillar plots:** Subject random effects (BLUPs) from GLMM

## Quick Start

```powershell
python -X utf8 -u scripts/run_posthoc_stats.py \
  --run-dir "results\runs\<your_run_directory>" \
  --alpha 0.05 \
  --multitest fdr \
  --glmm \
  --forest
```

**Expected time:** 1-5 minutes

## Command-Line Arguments

- `--run-dir` (str, required): Path to completed training run
- `--alpha` (float, optional): Significance threshold (default: 0.05)
- `--multitest` (str, optional): Multiple testing correction (`fdr` or `none`, default: `fdr`)
- `--glmm` (flag, optional): Run GLMM analysis (requires R with lme4)
- `--forest` (flag, optional): Generate forest plots
- `--chance-rate` (float, optional): Chance rate (auto-detected if omitted)

## Outputs

All outputs written to `<run_dir>/stats/`:

### Group-Level Statistics

**`group_stats.json`:**
```json
{
  "accuracy": {
    "mean": 0.556,
    "ci_lower": 0.521,
    "ci_upper": 0.591,
    "perm_pvalue": 0.005
  },
  "macro_f1": {
    "mean": 0.554,
    "ci_lower": 0.519,
    "ci_upper": 0.589,
    "perm_pvalue": 0.005
  },
  "chance_rate": 0.5,
  "n_subjects": 24
}
```

**Interpretation:**
- `mean`: Cross-fold mean performance
- `ci_lower/upper`: 95% bootstrap confidence interval
- `perm_pvalue`: Empirical p-value from permutation test (if available)
- p < 0.05: Significant above-chance performance

### Subject-Level Significance

**`per_subject_significance.csv`:**
```
subject_id,n_trials,accuracy,binomial_pvalue,adjusted_pvalue,significant
1,127,0.614,0.023,0.092,false
2,133,0.692,0.001,0.008,true
3,125,0.520,0.421,0.601,false
...
```

**Columns:**
- `binomial_pvalue`: One-tailed binomial test (H₀: accuracy = chance)
- `adjusted_pvalue`: FDR-corrected p-value (Benjamini-Hochberg)
- `significant`: True if adjusted_pvalue < alpha

**Interpretation:** Proportion of subjects significantly above chance.

**`per_subject_summary.json`:**
```json
{
  "n_subjects_above_chance": 18,
  "prop_subjects_above_chance": 0.75,
  "alpha": 0.05,
  "multitest_correction": "fdr"
}
```

### Forest Plot

**`per_subject_forest.png`:**
- X-axis: Accuracy (0-1)
- Y-axis: Subject IDs
- Points: Per-subject accuracy
- Error bars: Wilson 95% confidence intervals (binomial)
- Vertical line: Chance rate
- Color: Green (significant) vs gray (non-significant)

**Use case:** Visualize inter-subject variability, identify outliers.

### GLMM Analysis (Optional)

**`glmm_summary.json`:**
```json
{
  "fixed_effect_intercept": {
    "estimate_logodds": 0.224,
    "ci_lower_logodds": 0.085,
    "ci_upper_logodds": 0.363,
    "pvalue": 0.002,
    "estimate_prob": 0.556,
    "ci_lower_prob": 0.521,
    "ci_upper_prob": 0.590
  },
  "random_effect_sd": 0.312,
  "n_subjects": 24,
  "n_observations": 3024
}
```

**Interpretation:**
- `estimate_prob`: Population-level mean accuracy (on probability scale)
- `ci_lower/upper_prob`: 95% CI for population mean
- `pvalue`: Test of H₀: logit(accuracy) = logit(chance)
- `random_effect_sd`: Between-subject variability (log-odds scale)

**`glmm_intercept_effect.png`:**
- Dual-scale plot (log-odds and probability)
- Fixed-effect estimate + 95% CI
- Horizontal line at chance rate

**`glmm_caterpillar.png`:**
- X-axis: Predicted accuracy (probability scale)
- Y-axis: Subject IDs (sorted by BLUP)
- Points: Subject-specific predictions (BLUPs = fixed effect + random effect)
- Error bars: Subject-specific 95% CIs
- Vertical line: Population mean

**Use case:** Visualize subject heterogeneity, identify high/low performers.

### Permutation Test Outputs (If Available)

**`perm_density_acc.png` and `perm_density_macro_f1.png`:**
- Density plot of null distribution (from permutation test)
- Vertical line: Observed performance
- Shaded region: p < 0.05 threshold

**Interpretation:** Visual assessment of how far observed performance exceeds null.

### QQ Plot (FDR Diagnostic)

**`qq_pvalues_fdr.png`:**
- X-axis: Expected p-values under uniform null
- Y-axis: Observed p-values (per-subject)
- Diagonal line: Perfect null
- Horizontal line: BH threshold for FDR control

**Interpretation:**
- Points below diagonal: More significant than expected (true effects)
- Points above diagonal: Less significant (null or underpowered)

## Configuration Defaults

Defaults can be set in `configs/posthoc_defaults.yaml`:

```yaml
posthoc:
  alpha: 0.05
  multitest: fdr
  glmm: true
  forest: true
  chance_rate: null  # Auto-detect from task
```

**CLI overrides YAML:** Command-line arguments take precedence.

## Statistical Methods

### Binomial Test (Per-Subject)

**Null hypothesis:** Subject's accuracy = chance rate

**Test statistic:** Number of correct trials ~ Binomial(n_trials, chance_rate)

**P-value:** One-tailed (upper) probability

**Assumptions:**
- Trials are independent (may be violated with repeated measures)
- Chance rate is known (inferred from number of classes)

### FDR Correction (Benjamini-Hochberg)

**Purpose:** Control false discovery rate across multiple subject tests.

**Method:** Rank p-values, find largest k such that `p(k) ≤ (k/m) × α`.

**Effect:** More powerful than Bonferroni (fewer false negatives), but less conservative.

**When to use:** Testing many subjects (m > 10), expect some true effects.

**When to skip:** Single-subject analysis, or strict family-wise error rate control needed (use Bonferroni instead).

### Generalized Linear Mixed Model (GLMM)

**Model specification:**
```R
glmer(correct ~ 1 + (1|subject_id), family=binomial(link="logit"), data=trials)
```

**Fixed effect:** Intercept (population-level log-odds of correct response)

**Random effect:** Subject-specific intercept (captures between-subject variability)

**Advantages over t-test:**
- Accounts for variable trial counts per subject
- Models binomial outcomes directly (not averaged accuracy)
- Provides subject-specific predictions (BLUPs)

**Requirements:**
- R installed with `lme4` package
- `test_predictions_outer.csv` exists in run directory

**Fallback:** If R unavailable, GLMM analysis skipped (other analyses proceed).

### Wilson Score Interval (Forest Plot)

**Purpose:** Confidence intervals for binomial proportions (better than Wald for small n or extreme p).

**Formula:** Adjusted Wald interval with continuity correction.

**Properties:**
- Asymmetric (not centered on point estimate)
- Coverage closer to nominal 95% than normal approximation
- Handles p near 0 or 1 gracefully

## Interpretation Guidelines

### Group-Level Significance

**Accuracy significantly above chance?**
- Check `perm_pvalue` in `group_stats.json`
- If p < 0.05: Yes, group-level evidence for decoding
- If p ≥ 0.05: No group-level evidence (may still have subject-level effects)

**Caveat:** Group mean can be above chance even if only subset of subjects decode successfully.

### Subject-Level Reliability

**What proportion of subjects show significant decoding?**
- Check `prop_subjects_above_chance` in `per_subject_summary.json`
- High proportion (>75%): Strong subject-level reliability
- Medium proportion (50-75%): Moderate reliability, some subjects don't decode
- Low proportion (<50%): Weak reliability, decoding driven by outliers

**Use case:** Assess generalizability. High group accuracy + low subject-level reliability suggests a few strong performers drive mean.

### GLMM Insights

**Population-level inference:**
- Fixed effect p-value tests population mean against chance
- More robust than t-test on per-subject means (handles unequal trial counts)

**Subject heterogeneity:**
- Large `random_effect_sd`: High between-subject variability
- Small `random_effect_sd`: Homogeneous population
- Caterpillar plot shows which subjects deviate from population mean

## Common Issues

### R Not Found

**Symptom:** `GLMM analysis skipped: R not installed`

**Solution:** Install R and ensure it's in PATH, or omit `--glmm` flag.

**R installation:**
1. Download from https://cran.r-project.org/
2. Install `lme4`: `R -e 'install.packages("lme4")'`
3. Restart terminal (refresh PATH)

### Permutation P-Values Missing

**Symptom:** `perm_pvalue: null` in `group_stats.json`

**Cause:** Permutation test not run for this task.

**Solution:** Run permutation test (see [Workflows](WORKFLOWS.md#permutation-testing-workflow)), then re-run post-hoc stats.

### All Subjects Non-Significant

**Symptom:** `n_subjects_above_chance: 0`

**Causes:**
1. Model performs at chance (no decoding)
2. Too few trials per subject (underpowered binomial tests)
3. Alpha too strict (try alpha=0.1)

**Solutions:**
1. Check group-level accuracy in `outer_eval_metrics.csv` (should be >>chance)
2. Increase trial count (collect more data or use easier task)
3. Relax alpha (exploratory analysis)

### FDR Correction Too Conservative

**Symptom:** Many small p-values, but few pass FDR threshold.

**Cause:** BH procedure is conservative when many nulls are true.

**Solution:** Use `--multitest none` for uncorrected p-values (appropriate if pre-registered single test).

## Next Steps

- For running permutation tests, see [Workflows](WORKFLOWS.md#permutation-testing-workflow)
- For interpreting results in context, see main [README](../README.md#scientific-rigor--reproducibility)
- For subject performance breakdowns, see [CLI Reference](CLI_REFERENCE.md#scriptsanalyze_nloso_subject_performancepy)
