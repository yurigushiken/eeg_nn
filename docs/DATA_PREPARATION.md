# Data Preparation Guide

Complete guide to preparing EEG data for the neural decoding pipeline.

## Overview

The pipeline expects **materialized** `.fif` epoch files (one per subject), created by converting HAPPE-preprocessed EEGLAB `.set` files.

## Input Requirements

### HAPPE-Preprocessed EEG Data

**Expected location:**
```
data_input_from_happe/<dataset>/5 - processed .../*.set
```

**Format:** EEGLAB `.set` files (128-channel EEI system)

**Preprocessing applied (by HAPPE):**
- 0.3-30 Hz bandpass filter (FIR)
- Artifact rejection (bad channels, eye blinks, excessive noise)
- Spherical spline interpolation for bad channels
- Average reference

**Key properties:**
- Sampling rate: 250 Hz (typical)
- Channels: 128 (EGI HydroCel GSN)
- Epochs: 500ms (-100ms pre-stimulus, +400ms post-stimulus)

### Behavioral Data

**Expected location:**
```
data_behavior/data_UTF8/SubjectXX.csv
```

**Required columns:**
- `SubjectID`: Subject identifier (must match EEG filename)
- `Block`: Block number
- `Trial`: Trial number within block
- `Procedure`: Trial type (e.g., "TrialProc")
- `Condition`: Stimulus condition code
- `Target.ACC`: Accuracy (0/1)
- `Target.RT`: Reaction time (ms)

**Derived columns (computed by prep script):**
- `direction`: "increasing", "decreasing", or "no_change"
- `size`: "small", "large", or "crossover"
- `change_group`: Combination of direction + size

### Montage File

**Expected location:**
```
net/AdultAverageNet128_v1.sfp
```

**Format:** Standard 10-20 sensor position file (128-channel EGI)

**Purpose:** 3D electrode positions for topographic plotting

## Conversion Script

### Run Data Preparation

```powershell
python scripts/prepare_from_happe.py
```

**Expected runtime:** 5-15 minutes (depends on number of subjects)

**What it does:**
1. Loads HAPPE `.set` files for each subject
2. Aligns EEG epochs with behavioral CSV (trial-by-trial matching)
3. Removes `Condition==99` trials (practice/excluded trials)
4. Encodes labels for all tasks (cardinality, landing digit, etc.)
5. Attaches montage (`net/AdultAverageNet128_v1.sfp`)
6. Saves per-subject `.fif` epochs to `data_preprocessed/<dataset>/`

## Output Structure

**Location:**
```
data_preprocessed/<dataset>/sub-XX_preprocessed-epo.fif
```

**Format:** MNE-Python Epochs object (`.fif` format)

**Contents:**
- EEG data: Shape (n_epochs, n_channels, n_times) in microvolts (µV)
- Metadata: DataFrame with all behavioral columns + derived columns
- Channel info: Names, types, positions (from montage)
- Time axis: Sample times relative to stimulus onset

**Example metadata columns:**
```
SubjectID, Block, Trial, Condition, Target.ACC, Target.RT,
direction, size, change_group, landing_digit, cardinality, ...
```

## Data Model

### Epoch Structure

Each `.fif` file contains epochs for one subject:
- **Channels:** 128 EEG sensors (after non-scalp exclusion: ~100 channels)
- **Time samples:** 500ms at 250 Hz = 125 samples
- **Epochs:** Variable per subject (~270 trials, depends on accuracy)

### Label Encoding

Labels are encoded in metadata for each task:

**Cardinality tasks:**
- `cardinality_1_3`: Labels 0/1/2 → numerosities 1/2/3
- `cardinality_4_6`: Labels 0/1/2 → numerosities 4/5/6
- `cardinality_1_6`: Labels 0-5 → numerosities 1-6

**Landing digit tasks:**
- `landing_on_2_3`: Labels 0/1 → landing digits 2/3
- `landing_digit_1_3_within_small`: Labels 0/1/2 → landing digits 1/2/3

**Special columns:**
- `direction`: "increasing", "decreasing", "no_change"
- `size`: "small" (1-3), "large" (4-6), "crossover"

### Channel Selection

**Default policy:** Exclude 28 non-scalp channels (specified in `code/preprocessing/epoch_utils.py`)

**Non-scalp channels:**
- E17, E38, E43, E44, E48, E49, E56, E63, E68, E73, E81, E88, E94, E99, E107, E113, E119, E120, E125, E126, E127, E128, VEOGL, VEOGR, VEOGU, VEOGD, Cz_original, M1, M2

**Remaining:** ~100 scalp EEG channels (covers entire head)

**Override:** Use `include_channels` in config to specify custom subset.

## Data Alignment

### Epoch-Behavior Matching

**Strict alignment enforced:**
1. For each EEG epoch, find corresponding behavioral trial by `(SubjectID, Block, Trial)`
2. If no match found → raise error (no silent drops)
3. If duplicate matches → raise error (ambiguous trials)

**Rationale:** Prevents silent misalignment (e.g., epochs shifted by one trial).

**Handling mismatches:**
- Check behavioral CSV has all trials in EEG `.set`
- Check `Condition` column excludes practice trials (99)

### Condition Filtering

**Excluded conditions:**
- `Condition == 99`: Practice trials, not analyzed

**Included conditions:**
- All other conditions (numerical change detection trials)

**Task-specific filtering:**
- `ACC=1` tasks: Only correct responses (e.g., `landing_on_2_3`)
- `ALL` tasks: Include errors and no-change (e.g., `landing_digit_1_3_within_small_and_cardinality`)

## Common Issues

### FileNotFoundError: HAPPE Data

**Symptom:** `FileNotFoundError: data_input_from_happe/<dataset>/...`

**Solutions:**
1. Check HAPPE output path matches script expectations
2. Edit `scripts/prepare_from_happe.py` to update paths
3. Ensure HAPPE preprocessing completed successfully

### Behavior CSV Encoding Error

**Symptom:** `UnicodeDecodeError` when loading behavioral CSV

**Solution:** Ensure CSV is UTF-8 encoded. Convert with:
```powershell
Get-Content behavior.csv | Set-Content -Encoding UTF8 behavior_utf8.csv
```

### Montage File Missing

**Symptom:** `FileNotFoundError: net/AdultAverageNet128_v1.sfp`

**Solution:**
1. Check `net/` directory exists with montage file
2. If missing, obtain from EGI or MNE-Python default montages

**Consequence:** Topoplots unavailable in XAI analysis (other analyses proceed).

### Epoch-Behavior Mismatch

**Symptom:** `AssertionError: Epochs and behavior counts don't match`

**Causes:**
1. Behavioral CSV missing trials present in EEG
2. EEG missing trials present in behavior
3. Duplicate trials in behavioral CSV

**Solutions:**
1. Check HAPPE artifact rejection didn't remove trials
2. Check behavioral CSV has all blocks
3. Remove duplicate rows in behavioral CSV

### Empty .fif Files

**Symptom:** `.fif` files created but contain 0 epochs

**Causes:**
1. All trials excluded (e.g., all `Condition==99`)
2. All trials rejected by HAPPE
3. Task label encoding failed (no matching trials)

**Solutions:**
1. Check behavioral CSV has non-practice trials
2. Check HAPPE QC metrics (excessive rejection?)
3. Inspect metadata after loading .fif in Python

## Customizing Data Prep

### Editing Prep Script

**Location:** `scripts/prepare_from_happe.py`

**Common edits:**
1. **Change input paths:** Update `happe_dir`, `behavior_dir`
2. **Change filter settings:** Already done by HAPPE (0.3-30 Hz)
3. **Add custom metadata:** Compute derived columns, add to `metadata` DataFrame
4. **Custom epoch rejection:** Add trial-level exclusion criteria

**Example: Add custom column**
```python
# In prepare_from_happe.py, after behavior loading:
metadata['my_column'] = metadata['Condition'].apply(lambda x: custom_logic(x))
```

### Alternative Preprocessing

If using non-HAPPE preprocessing:

1. Ensure data is in MNE-compatible format (`.set`, `.fif`, `.edf`)
2. Match expected structure (epochs, metadata columns)
3. Attach montage for topoplots
4. Save as `.fif`: `epochs.save(output_path, overwrite=True)`

## Data Quality Checks

After running `prepare_from_happe.py`:

### Check Epoch Counts

```python
import mne

# Load one subject
epochs = mne.read_epochs("data_preprocessed/<dataset>/sub-01_preprocessed-epo.fif")

print(f"Number of epochs: {len(epochs)}")
print(f"Channels: {len(epochs.ch_names)}")
print(f"Samples: {len(epochs.times)}")
print(f"Sampling rate: {epochs.info['sfreq']} Hz")
```

**Expected:**
- Epochs: ~270 per subject (depends on accuracy)
- Channels: ~100 (after non-scalp exclusion)
- Samples: 125 (500ms at 250 Hz)

### Check Metadata

```python
print(epochs.metadata.columns)
print(epochs.metadata['Target.ACC'].value_counts())
print(epochs.metadata['direction'].value_counts())
```

**Expected columns:** `SubjectID`, `Target.ACC`, `direction`, `size`, `landing_digit`, etc.

### Visualize Epochs

```python
# Plot first 10 epochs
epochs[:10].plot(n_epochs=10, n_channels=30)

# Plot average ERP
epochs.average().plot()
```

**Look for:**
- Clear ERP components (P1, N1, P3b)
- No excessive artifacts (flat lines, extreme amplitudes)

## Next Steps

- For training with prepared data, see [Quick Start Guide](QUICK_START.md)
- For understanding data structure in training, see [Technical Details](TECHNICAL_DETAILS.md)
- For troubleshooting data issues, see [Troubleshooting Guide](TROUBLESHOOTING.md)
