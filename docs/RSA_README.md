# RSA Binary Matrix Workflow (Internal)

This document captures everything we need to reproduce the RSA binary matrix analyses end‑to‑end. It assumes you already cloned `eeg_nn`, downloaded the study materials, and are working inside the managed Conda env (`conda activate eegnex-env`). Nothing here is meant for publication—think of it as our lab notebook for the RSA pipeline.

---

## 1. What This Workflow Does

We train fixed EEGNeX models on every cross-digit pairing from the cardinality set {11, 22, 33, 44, 55, 66}. Each run performs subject-aware outer evaluation, logs `outer_eval_metrics.csv`, and the helper scripts aggregate/visualize the results as a representational dissimilarity matrix (RDM) plus MDS plots and subject-level statistics.

| Component | Purpose |
|-----------|---------|
| `scripts/run_rsa_matrix.py` | Launches training for all pairs × seeds using a base YAML config. Optionally auto-generates figures. |
| `scripts/compile_rsa_results.py` | Crawls run folders and writes a master CSV (`rsa_results_master.csv`) with every OVERALL, fold, and true per-subject metric (using `test_predictions_outer.csv`). |
| `scripts/visualize_rsa.py` | Produces the Columbia-blue RDM heatmap and the publication-ready MDS scatter. |
| `scripts/analyze_rsa_stats.py` | Runs per-pair t-tests over subject means (seed-averaged) with Holm correction against a configurable baseline (default 50%). |

---

## 2. Configurations & Variants

We keep configs in `configs/tasks/`:

| Config | Description |
|--------|-------------|
| `rsa_binary.yaml` | Default workflow—GroupKFold with `n_folds: 6` and seeds `[42, 43, 44]`. Fast enough for daily iterations. |
| `rsa_binary_loso10.yaml` | LOSO version with 10 seeds `[41…50]` for publication-grade robustness. |

Both configs share the same locked hyperparameters (from the best `cardinality_3_vs_4` Optuna run) and disable augmentation for reproducibility. They also bake in visualization defaults:

```yaml
rsa_visualize: true
rsa_visualize_output_dir: results/figures/rsa_matrix_v1
rsa_visualize_prefix: rsa_matrix_v1
rsa_visualize_metric: Accuracy
```

You can override any of these on the CLI (see below).

---

## 3. Launching Training

```powershell
conda activate eegnex-env
cd D:\eeg_nn

python -X utf8 -u scripts/run_rsa_matrix.py `
  --config configs/tasks/rsa_binary.yaml `
  --output-dir results\runs\rsa_matrix_v1
```

Key notes:

* **Run layout** – Each seed/pair combination lands in `results\runs\rsa_matrix_v1\<timestamp>_rsa_<AA>v<BB>_seed_<SS>/`. The directory contains logs, checkpoints, `outer_eval_metrics.csv`, and plots from the standard training loop.
* **Summary CSV** – The script also writes `results\runs\rsa_matrix_v1\rsa_matrix_results.csv` (overall metrics only). We still prefer the master CSV produced in the next step.
* **Conditions filter** – Use `--conditions 11 22 33` if you only want a subset (helpful for smoke tests).
* **Visualization hooks** – If the config sets `rsa_visualize: true`, the script will call `visualize_rsa.py` once the master CSV exists (after you run the compile step). Otherwise you can manually run the visualization command later.
* **Alternate config** – Swap in `--config configs/tasks/rsa_binary_loso10.yaml` for the LOSO + 10-seed campaign. Make sure to change `--output-dir` so you don’t mix datasets.

---

## 4. Compile the Master Dataset

After all runs finish:

```powershell
python -X utf8 -u scripts/compile_rsa_results.py `
  --runs-dir results\runs\rsa_matrix_v1 `
  --output results\runs\rsa_matrix_v1\rsa_results_master.csv
```

* The script parses folder names (`rsa_<AA>v<BB>_seed_<SS>`), ingests `outer_eval_metrics.csv`, **and** derives per-subject accuracies from each run's `test_predictions_outer.csv` (fallback `test_predictions.csv`).
* Each row includes `ClassA`, `ClassB`, `Seed`, `Fold`, `Subject`, `RecordType`, `Accuracy`, `MacroF1`, and `MinClassF1`.
* OVERALL rows capture ensemble metrics; fold rows retain the held-out quartet ID string; subject rows set `RecordType: subject` and one subject ID per line. The stats scripts refuse to run unless these subject rows exist, so do not delete the predictions CSVs.

If you rerun training or add new seeds later, rerun this command—the master CSV is our single source of truth for downstream analysis.

---

## 5. Visualization (RDM + MDS)

```powershell
python -X utf8 -u scripts/visualize_rsa.py `
  --csv results\runs\rsa_matrix_v1\rsa_results_master.csv `
  --subject OVERALL `
  --output-dir results\figures\rsa_matrix_v1 `
  --prefix rsa_matrix_v1
```

Outputs:

1. `rsa_matrix_v1_rdm_heatmap.png` – Lower triangle RDM, 50–80% accuracy scale, cardinality labels on both axes, redundant upper triangle masked.
2. `rsa_matrix_v1_mds.png` – 2D projection using our corrected distance metric (`distance = accuracy - 50`, clipped at 0). The x-axis is fixed to ±7.5 with 2.5-unit ticks so figures are consistent across runs. y-range auto-adjusts with padding so labels remain inside the frame.

Additional flags:

* `--metric MacroF1` to visualize a different column.
* `--subject 13` to isolate a single subject (subject IDs must match `Subject` values in the master CSV).
* When run via `run_rsa_matrix.py --visualize`, `--viz-output-dir`, `--viz-prefix`, and `--viz-metric` map directly to these arguments.

---

## 6. Subject-Level Statistics

```powershell
python -X utf8 -u scripts/analyze_rsa_stats.py `
  --csv results\runs\rsa_matrix_v1\rsa_results_master.csv `
  --baseline 50 `
  --output results\runs\rsa_matrix_v1\stats_summary.csv
```

What you get per pair:

* `n_subjects` – number of unique participants contributing to that pairing (after averaging each participant across seeds; 24 for the current dataset when every subject has data).
* `mean_accuracy` and `std_accuracy` – subject means in percentage units.
* `t_stat`, `p_value`, `p_value_holm` – one-sample t-test vs. the specified baseline plus Holm-Bonferroni correction across all 15 pairings. If SciPy is unavailable, the script falls back to a normal approximation.

Use the CSV for quick tables or import it into notebooks for visualization.

---

## 7. Typical Output Tree

```
results/
└─ runs/
   └─ rsa_matrix_v1/
      ├─ rsa_matrix_results.csv
      ├─ rsa_results_master.csv
      ├─ stats_summary.csv
      ├─ 20251204_195154_rsa_11v22_seed_42/
      │   ├─ outer_eval_metrics.csv
      │   ├─ logs/…
      │   └─ plots/…
      └─ figures/
          ├─ rsa_matrix_v1_rdm_heatmap.png
          └─ rsa_matrix_v1_mds.png
```

Adjust names if you run LOSO (`rsa_matrix_loso10` etc.), but keep the structure consistent so automation scripts find what they need.

---

## 8. Tips & Troubleshooting

* **Missing `outer_eval_metrics.csv`** – The compile script skips directories without the file. If a run crashed mid-way it won’t show up; rerun `run_rsa_matrix.py` for that pair/seed.
* **Mismatched folder names** – `compile_rsa_results.py` expects `rsa_<AA>v<BB>_seed_<SS>`. If you copy or rename folders, keep this format.
* **Visuals not updating** – Delete/rename the old figures and rerun `visualize_rsa.py`; it overwrites files but PowerShell might cache preview thumbnails.
* **Changing metrics** – Any column in the master CSV can be visualized or used in stats. Make sure the column exists (e.g., `MinClassF1`).
* **Adding more seeds** – Duplicate the base YAML, tweak `seeds`, `n_folds`, and `rsa_visualize_*`, give the run a new output directory, and follow the same compile/visualize steps.
* **Performance sanity checks** – Use `stats_summary.csv` to verify subject counts (24 for either GroupKFold or LOSO on this dataset). If you see `n_subjects` < expected, one or more predictions CSVs may be missing; re-run the compiler once the runs are complete.

---

## 9. Quick Reference Commands

| Stage | Command |
|-------|---------|
| Training (default) | `python -X utf8 -u scripts/run_rsa_matrix.py --config configs/tasks/rsa_binary.yaml --output-dir results\runs\rsa_matrix_v1` |
| Training (LOSO 10 seeds) | `python -X utf8 -u scripts/run_rsa_matrix.py --config configs/tasks/rsa_binary_loso10.yaml --output-dir results\runs\rsa_matrix_loso10` |
| Compile master CSV | `python -X utf8 -u scripts/compile_rsa_results.py --runs-dir <run_dir> --output <run_dir>\rsa_results_master.csv` |
| Visualize | `python -X utf8 -u scripts/visualize_rsa.py --csv <run_dir>\rsa_results_master.csv --subject OVERALL --output-dir <run_dir>\figures --prefix <tag>` |
| Stats | `python -X utf8 -u scripts/analyze_rsa_stats.py --csv <run_dir>\rsa_results_master.csv --baseline 50 --output <run_dir>\stats_summary.csv` |
| Tables (LaTeX/Markdown) | `python -X utf8 -u scripts/generate_rsa_tables.py --csv <run_dir>\stats_summary.csv --output-dir <run_dir>\tables` |

---

Keep this README updated whenever we tweak configs, add new metrics, or change naming conventions. It should always reflect the exact process we expect new lab members to follow.
