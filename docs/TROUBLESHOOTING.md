# Troubleshooting Guide

Common issues, error messages, and solutions for the EEG neural decoding pipeline.

## Installation & Environment

### `ModuleNotFoundError: No module named 'mne'`

**Cause:** Conda environment not activated or incorrectly installed.

**Solution:**
```powershell
conda activate eegnex-env
```

If environment doesn't exist:
```powershell
conda env create -f environment.yml
```

### `ImportError: DLL load failed` (Windows)

**Cause:** Missing Visual C++ Redistributable or CUDA libraries.

**Solutions:**
1. Install Visual C++ Redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Ensure CUDA toolkit matches PyTorch version (check `environment.yml`)
3. Reinstall PyTorch:
   ```powershell
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

### `playwright install` fails

**Cause:** Network issues or missing dependencies.

**Solution:**
```powershell
# Try with specific browser
playwright install chromium

# Or install all
playwright install
```

If fails, PDF generation unavailable (HTML reports still work).

---

## Data Preparation

### `FileNotFoundError: data_input_from_happe/...`

**Cause:** HAPPE output path doesn't match script expectations.

**Solution:** Edit `scripts/prepare_from_happe.py` and update paths:
```python
happe_dir = "path/to/your/happe/output"
behavior_dir = "path/to/your/behavior/csvs"
```

### `UnicodeDecodeError` when loading behavioral CSV

**Cause:** CSV not UTF-8 encoded.

**Solution:** Convert CSV to UTF-8:
```powershell
Get-Content behavior.csv | Set-Content -Encoding UTF8 behavior_utf8.csv
```

### `AssertionError: Epochs and behavior counts don't match`

**Causes:**
1. Behavioral CSV missing trials present in EEG
2. EEG missing trials present in behavior (artifact rejection)
3. Duplicate trials in behavioral CSV

**Solutions:**
1. Check HAPPE QC: Were many trials rejected?
2. Check behavioral CSV: All blocks present?
3. Remove duplicate rows in behavioral CSV
4. Inspect one subject manually:
   ```python
   import mne
   epochs = mne.read_epochs("data_preprocessed/.../sub-01_preprocessed-epo.fif")
   print(f"Epochs: {len(epochs)}")
   print(f"Behavior rows: {len(epochs.metadata)}")
   ```

### Empty `.fif` files (0 epochs)

**Causes:**
1. All trials excluded (`Condition==99`)
2. All trials rejected by HAPPE (100% artifact rate)
3. Task label encoding failed

**Solutions:**
1. Check `Condition` column in behavioral CSV (should have non-99 values)
2. Check HAPPE QC metrics (rejection rate < 50% typical)
3. Debug `prepare_from_happe.py`: Add print statements before/after filtering

---

## Training Errors

### `KeyError: 'seed'`

**Cause:** `seed` parameter missing from config (constitutional requirement).

**Solution:** Add to your config YAML:
```yaml
seed: 42
```

### `KeyError: 'optuna_objective'` (during Optuna search)

**Cause:** Objective metric not specified in search controller.

**Solution:** Add to `stepX_search.yaml`:
```yaml
optuna_objective: composite_min_f1_plur_corr
composite_min_f1_weight: 0.65  # If using composite
```

### `KeyError: 'outer_eval_mode'`

**Cause:** Evaluation mode not specified (constitutional requirement).

**Solution:** Add to config:
```yaml
outer_eval_mode: ensemble  # or "refit"
```

### `CUDA out of memory`

**Causes:**
1. Batch size too large
2. Model too large (too many channels/parameters)
3. GPU too small

**Solutions:**
1. Reduce batch size:
   ```powershell
   python train.py ... --set batch_size=16
   ```
2. Use fewer channels (enable Cz-ring):
   ```yaml
   cz_step: 3  # Keep 60% of channels
   ```
3. Use smaller model:
   ```yaml
   F1: 8  # Instead of 32
   ```
4. Use CPU (slow):
   ```powershell
   $env:CUDA_VISIBLE_DEVICES=""
   python train.py ...
   ```

### `AssertionError: Subject overlap detected between train and test`

**Cause:** Bug in split generation (should never happen).

**Solution:** This is a safeguard. If triggered:
1. Check `splits_indices.json` for overlap
2. Report issue on GitHub with config file

### `AssertionError: Insufficient subjects for inner_n_folds=5`

**Cause:** Not enough subjects in outer-train to split into 5 inner folds.

**Solution:** Reduce `inner_n_folds`:
```yaml
inner_n_folds: 2  # Minimum
```

Or use more subjects (increase N).

---

## Optuna Issues

### `DuplicateStudyError: Study 'xxx' already exists`

**Cause:** Running same study name twice.

**Solutions:**
1. **Continue study:** Omit or increase `--trials` (Optuna resumes automatically)
2. **Start new study:** Change `optuna_study_name` in controller YAML
3. **Delete old study:**
   ```powershell
   Remove-Item "results\optuna\<study_name>.db"
   ```

### Optuna study not found when refreshing

**Cause:** No `.db` files in `results/optuna/` or wrong directory.

**Solution:**
1. Check `results/optuna/` contains `*.db` files
2. Run at least one Optuna search first

### All trials pruned immediately

**Causes:**
1. Search space too narrow (all configurations bad)
2. `n_warmup_steps` too low (pruning too aggressive)
3. Bug in objective computation

**Solutions:**
1. Widen search space (more hyperparameter options)
2. Increase `n_warmup_steps` to 10-20:
   ```yaml
   optuna_pruner_kwargs:
     n_warmup_steps: 15
   ```
3. Check objective is computing correctly (inspect trial logs)

### Top-3 report missing

**Causes:**
1. Fewer than 3 completed trials
2. Playwright not installed (PDF generation failed, but HTML should exist)

**Solutions:**
1. Run more trials (need ≥3)
2. Check `results/optuna/<study>_top3_report.html` (HTML always generated)
3. Install Playwright for PDF:
   ```powershell
   playwright install
   ```

---

## XAI Analysis Issues

### No topomaps generated

**Symptom:** Heatmaps exist, but no `.png` topomaps.

**Causes:**
1. Montage file missing (`net/AdultAverageNet128_v1.sfp`)
2. Channel names don't match montage
3. MNE can't find channels in montage

**Solutions:**
1. Check montage exists:
   ```powershell
   ls net/AdultAverageNet128_v1.sfp
   ```
2. Verify channel names match:
   ```python
   import mne
   epochs = mne.read_epochs("data_preprocessed/.../sub-01_preprocessed-epo.fif")
   print(epochs.ch_names[:10])  # Should match montage (e.g., E1, E2, ...)
   ```
3. Check console output for montage warnings

### Time-frequency analysis skipped

**Symptom:** "Signal too short for lowest frequency" warning.

**Cause:** Epoch duration < longest wavelet period (e.g., 4 Hz needs ~250ms).

**Solutions:**
1. Use longer epochs:
   ```yaml
   crop_ms: [0, 800]  # Instead of [0, 400]
   ```
2. Remove lowest frequency:
   ```yaml
   tf_morlet_freqs: [8, 13, 30]  # Remove 4 Hz
   ```

### PDF report generation failed

**Symptom:** HTML exists, but no PDF.

**Cause:** Playwright not installed.

**Solution:**
```powershell
playwright install
```

HTML report always generated (PDF is optional).

### Attributions look random/noisy

**Symptoms:** No clear spatial or temporal structure in heatmaps.

**Causes:**
1. Model performance at chance (no learned patterns)
2. Very few test trials (noisy estimates)
3. Model overfitting

**Solutions:**
1. Check accuracy in `outer_eval_metrics.csv` (should be >>chance)
2. Use more test trials (more subjects or GroupKFold instead of LOSO)
3. Re-run hyperparameter search with better regularization:
   ```yaml
   dropout: 0.5  # Increase
   weight_decay: 0.001  # Increase
   ```

---

## Post-Hoc Statistics Issues

### R not found (GLMM analysis skipped)

**Symptom:** `GLMM analysis skipped: R not installed`

**Solutions:**
1. Install R from https://cran.r-project.org/
2. Install `lme4`:
   ```R
   R -e 'install.packages("lme4", repos="http://cran.r-project.org")'
   ```
3. Restart terminal (refresh PATH)
4. Or omit `--glmm` flag (other analyses proceed)

### All subjects non-significant

**Symptom:** `n_subjects_above_chance: 0`

**Causes:**
1. Model performs at chance (no decoding)
2. Too few trials per subject (underpowered)
3. Alpha too strict

**Solutions:**
1. Check group accuracy (should be >55% for binary task)
2. Collect more data or use easier task
3. Relax alpha:
   ```powershell
   python run_posthoc_stats.py ... --alpha 0.1
   ```

### Permutation p-values missing

**Symptom:** `perm_pvalue: null` in `group_stats.json`

**Cause:** Permutation test not run.

**Solution:** Run permutation test first (see [Workflows](WORKFLOWS.md#permutation-testing-workflow)).

---

## Performance Issues

### Optuna search extremely slow

**Symptom:** Each trial takes 30+ minutes.

**Causes:**
1. Dataset caching disabled (reloading .fif every trial)
2. Too many folds (LOSO with 24 subjects = 24 folds)
3. Too many epochs per trial

**Solutions:**
1. **Enable caching (CRITICAL):**
   ```yaml
   dataset_cache_memory: true
   ```
   This alone gives 10-100× speedup.

2. Use fewer folds for search:
   ```yaml
   n_folds: 3  # Instead of null (LOSO)
   inner_n_folds: 2
   ```

3. Reduce epochs for search:
   ```yaml
   epochs: 50  # Instead of 100
   early_stop: 10
   ```

### Training hangs at start

**Symptom:** Script starts, prints config, then freezes.

**Causes:**
1. Loading large .fif files (first time without cache)
2. Deadlock in DataLoader (rare)

**Solutions:**
1. Wait 1-2 minutes (first load is slow)
2. Check console for progress messages
3. If truly hung (>5 min), Ctrl+C and retry with:
   ```yaml
   num_workers: 0  # Disable DataLoader multiprocessing
   ```

### Out of memory (RAM, not CUDA)

**Symptom:** `MemoryError` or system freeze.

**Cause:** Dataset caching uses too much RAM (>16 GB).

**Solutions:**
1. Disable caching:
   ```yaml
   dataset_cache_memory: false
   ```
2. Use fewer channels:
   ```yaml
   cz_step: 2  # Keep only 40% of channels
   ```
3. Close other applications
4. Upgrade RAM (16 GB → 32 GB)

---

## Git & Version Control

### Merge conflict in `README.md`

**Cause:** Documentation updated in both branches.

**Solution:**
1. Accept incoming changes (new streamlined README)
2. Old README content preserved in `docs/` files

### Accidentally committed large files

**Symptom:** `.fif` or `.db` files in Git history (repo bloat).

**Solution:**
1. Add to `.gitignore`:
   ```
   data_preprocessed/
   results/optuna/*.db
   ```
2. Remove from history (if already committed):
   ```powershell
   git rm --cached data_preprocessed/*.fif
   git commit -m "Remove large files from tracking"
   ```

---

## Debugging Tips

### Enable verbose logging

Most scripts support verbose mode:
```powershell
python train.py ... --verbose
```

Or set environment variable:
```powershell
$env:PYTHONVERBOSE=1
```

### Inspect resolved config

Every run writes `resolved_config.yaml`. Check it to see final merged config:
```powershell
cat results\runs\<run_dir>\resolved_config.yaml
```

### Check JSONL runtime log

All training events logged to `logs/runtime.jsonl`:
```powershell
cat results\runs\<run_dir>\logs\runtime.jsonl
```

Look for:
- Fold boundaries
- Class weights
- Split exports
- Error messages

### Test with minimal config

Strip config to bare minimum to isolate issue:
```yaml
seed: 42
outer_eval_mode: ensemble
optuna_objective: inner_mean_macro_f1
n_folds: 2  # Minimal
inner_n_folds: 2
epochs: 5  # Very short
batch_size: 16
lr: 0.001
```

If this works, gradually add back complexity.

### Use Python debugger

Add breakpoint in code:
```python
import pdb; pdb.set_trace()
```

Then run normally. Script will pause at breakpoint.

---

## Getting Help

### Before Opening Issue

1. Check this troubleshooting guide
2. Check relevant documentation:
   - [Quick Start](QUICK_START.md) - Basic usage
   - [Configuration](CONFIGURATION.md) - Parameter issues
   - [CLI Reference](CLI_REFERENCE.md) - Command syntax
3. Search existing GitHub issues: https://github.com/yourusername/eeg_nn/issues

### When Opening Issue

Include:
1. **Error message** (full traceback)
2. **Command used** (exact command with all arguments)
3. **Config file** (attach YAML or paste relevant sections)
4. **Environment info:**
   ```powershell
   conda list
   python --version
   nvidia-smi  # If CUDA issue
   ```
5. **Reproducible example** (minimal config that triggers issue)

### Contact

For questions or collaboration:
- Open issue on GitHub
- Email: mkg2145@tc.columbia.edu

---

## Known Limitations

### Windows-Specific Issues

**PowerShell line continuation:** Use backtick `` ` ``, ensure no trailing spaces.

**Path separators:** Use forward slashes `/` or double backslashes `\\` in Python args.

**UTF-8 encoding:** Always use `-X utf8` flag:
```powershell
python -X utf8 -u train.py ...
```

### Platform Compatibility

**Tested on:**
- Windows 10/11 (primary development platform)
- Linux (Ubuntu 20.04+)

**Not tested:**
- macOS (should work with minor path adjustments)
- Windows 7 (not supported)

### CUDA Determinism Trade-Offs

**Strict determinism enabled by default:**
- `torch.backends.cudnn.deterministic = True`
- `torch.use_deterministic_algorithms(True)`

**Effect:** ~10-30% slower training (deterministic algorithms less optimized).

**Disable for speed (non-reproducible):**
```yaml
# Not currently configurable, would require code changes
```

---

## Next Steps

- For detailed explanations, see [Technical Details](TECHNICAL_DETAILS.md)
- For understanding repository organization, see [Architecture](ARCHITECTURE.md)
- For common workflows, see [Workflows](WORKFLOWS.md)
