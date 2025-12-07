# Time-Resolved RSA: Implementation Guide

## Overview

Time-resolved RSA extends the static RSA analysis with a temporal dimension, using a sliding window approach to track when numerosity representations emerge during the EEG epoch.

## Methodology

**Deep Sliding Window Approach:**
- **Window size**: 50ms (≈12-13 samples at 250Hz)
- **Stride**: 20ms (≈5 samples, 75% overlap)
- **Epoch duration**: 500ms (0-496ms after cropping)
- **Result**: 23 time windows per epoch

**Temporal Windows:**
```
Window  1: t0-50ms
Window  2: t20-70ms
Window  3: t40-90ms
...
Window 23: t440-490ms
```

## Implementation

### Files Created

1. **[tests/test_temporal_decoding.py](../tests/test_temporal_decoding.py)** - 11 comprehensive tests
   - Window generation logic
   - Configuration handling
   - Results aggregation
   - State tracking (resumable execution)

2. **[scripts/run_temporal_decoding.py](../scripts/run_temporal_decoding.py)** - Main execution script
   - Mirrors structure of `run_rsa_matrix.py`
   - Triple loop: seeds × pairs × time windows
   - Resumable via state tracking
   - Outputs CSV with TimeWindow_Start/End/Center columns

3. **[configs/tasks/rsa_temporal.yaml](../configs/tasks/rsa_temporal.yaml)** - Configuration
   - Temporal parameters (epoch_ms, window_ms, stride_ms)
   - Inherits settings from rsa_binary.yaml
   - Configured for pilot run (1 seed, 1 pair)

### Test Results

All 11 tests pass:
```
✓ test_temporal_window_generation
✓ test_temporal_window_generation_edge_cases
✓ test_temporal_window_naming
✓ test_temporal_config_modification
✓ test_temporal_results_aggregation
✓ test_temporal_csv_header
✓ test_temporal_run_directory_naming
✓ test_temporal_state_tracking
✓ test_temporal_completion_tracking
✓ test_temporal_window_sample_calculation
✓ test_temporal_results_sorting
```

## Running the Analysis

### Pilot Run (Recommended First Step)

**Single pair (1 vs 2), 1 seed, 23 windows = 23 total runs**

```bash
conda activate eegnex-env

python scripts/run_temporal_decoding.py \
  --config configs/tasks/rsa_temporal.yaml \
  --conditions 11 22 \
  --output-dir results/runs/rsa_temporal_pilot
```

**Expected:**
- Runtime: ~5-6 hours
- Output: `results/runs/rsa_temporal_pilot/rsa_temporal_results.csv`
- 23 rows (one per time window)
- Columns: ClassA, ClassB, Seed, TimeWindow_Start, TimeWindow_End, TimeWindow_Center, Accuracy, MacroF1, MinClassF1

### Full Run (After Pilot Validation)

**All pairs (15), 3 seeds, 23 windows = 1,035 runs**

1. Update `configs/tasks/rsa_temporal.yaml`:
   ```yaml
   seeds: [42, 43, 44]  # Change from [42]
   # Remove --conditions flag to use all 6 numerosities
   ```

2. Run:
   ```bash
   python scripts/run_temporal_decoding.py \
     --config configs/tasks/rsa_temporal.yaml \
     --output-dir results/runs/rsa_temporal_v1
   ```

**Expected:**
- Runtime: ~250 hours (can be parallelized across windows or pairs)
- Output: `results/runs/rsa_temporal_v1/rsa_temporal_results.csv`

### Resume Interrupted Runs

The script supports resumable execution:

```bash
python scripts/run_temporal_decoding.py \
  --config configs/tasks/rsa_temporal.yaml \
  --output-dir results/runs/rsa_temporal_pilot \
  --resume
```

## Output Structure

### Directory Layout
```
results/runs/rsa_temporal_pilot/
├── .rsa_temporal_resume_state.json          # State for resuming
├── rsa_temporal_results.csv                 # Aggregated results
├── 20250107_120000_rsa_11v22_seed_42_t0-50ms/
│   ├── outer_eval_metrics.csv
│   ├── test_predictions.csv
│   └── ...
├── 20250107_120100_rsa_11v22_seed_42_t20-70ms/
└── ...
```

### Results CSV Format

| ClassA | ClassB | Seed | TimeWindow_Start | TimeWindow_End | TimeWindow_Center | Accuracy | MacroF1 | MinClassF1 |
|--------|--------|------|------------------|----------------|-------------------|----------|---------|------------|
| 11     | 22     | 42   | 0                | 50             | 25                | 62.5     | 60.2    | 58.1       |
| 11     | 22     | 42   | 20               | 70             | 45                | 65.3     | 63.1    | 61.2       |
| ...    | ...    | ...  | ...              | ...            | ...               | ...      | ...     | ...        |

## Visualization (After Pilot)

After the pilot run completes, we'll create:

1. **Emergence curves**: Accuracy vs time for pair 1v2
2. **Peak timing**: When does decodability peak?
3. **Baseline comparison**: Compare to chance (50%)

## Scientific Rationale

This approach allows us to answer:
- **When** do numerosity representations emerge in the EEG signal?
- Do small (1-3) and large (4-6) numerosities have different temporal dynamics?
- Does the pixel confound emerge at a different latency than cognitive structure?
- Can we identify critical time windows for numerical processing?

## Computational Cost

| Configuration | Runs | Estimated Time |
|---------------|------|----------------|
| Pilot (1 pair, 1 seed) | 23 | ~6 hours |
| Single pair (1 pair, 3 seeds) | 69 | ~18 hours |
| Full matrix (15 pairs, 3 seeds) | 1,035 | ~260 hours |

**Mitigation strategies:**
- Start with pilot to validate
- Parallelize across windows (each window is independent)
- Use cluster computing for full run
- Leverage resume functionality for fault tolerance

## Related Work

- **Cichy et al. (2014)** - Nature Neuroscience: Time-resolved RSA for visual object recognition
- **King & Dehaene (2014)** - Trends in Cognitive Sciences: Temporal generalization decoding
- **Caucheteux et al. (2021)** - Nature Neuroscience: Deep language models and brain encoding

## Next Steps

1. ✅ Run pilot (1 seed, 1 pair)
2. Validate results and visualize emergence curve
3. If successful, expand to more pairs or seeds
4. Apply same temporal analysis to confound control (partial correlation over time)
5. Create publication-ready temporal RDM movies

---

**Status**: Implementation complete, all tests passing, ready for pilot run.
**Contact**: This implementation follows TDD principles and mirrors the structure of `run_rsa_matrix.py` for maintainability.
