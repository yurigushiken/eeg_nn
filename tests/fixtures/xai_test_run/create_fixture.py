"""
Create minimal test fixture for XAI integration tests.

Generates:
- 2 subjects with 3 classes (cardinality: 1, 2, 3)
- Minimal EEG data (8 channels, 100 time points, ~500 Hz)
- Fold checkpoints (2 folds)
- summary.json with configuration
"""

import json
import sys
import numpy as np
import mne
import torch
from pathlib import Path

# Add project root to path so we can import code.model_builders
proj_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(proj_root))

# Configuration
FIXTURE_DIR = Path(__file__).parent
DATA_DIR = FIXTURE_DIR / "data"
CKPT_DIR = FIXTURE_DIR / "ckpt"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

# Dataset parameters
# Use standard EEG channel names that will match montage
# This ensures checkpoints match the filtered dataset
N_CHANNELS = 8
N_TIMES = 100
SFREQ = 500  # Hz
N_SUBJECTS = 2
TRIALS_PER_SUBJECT_PER_CLASS = 3
N_CLASSES = 3

# Standard 10-20 channel names (these will survive montage filtering)
CH_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4']

# Time vector
times = np.arange(N_TIMES) / SFREQ  # in seconds

print("Creating minimal test fixture...")

# Create .fif files for each subject
for subj_id in [1, 2]:
    print(f"Creating subject {subj_id:02d}...")
    
    # Generate random EEG data for this subject
    n_epochs = TRIALS_PER_SUBJECT_PER_CLASS * N_CLASSES  # 3 trials/class * 3 classes = 9 trials
    data = np.random.randn(n_epochs, N_CHANNELS, N_TIMES) * 1e-6  # microvolts in V
    
    # Create MNE Info object
    info = mne.create_info(
        ch_names=CH_NAMES,
        sfreq=SFREQ,
        ch_types=['eeg'] * N_CHANNELS
    )
    
    # Create metadata with cardinality labels (Condition column for label_fn)
    # For cardinality_1_3 task, valid conditions are 11, 22, 33 (same-digit pairs)
    metadata = []
    for class_idx in range(N_CLASSES):
        condition = [11, 22, 33][class_idx]  # 11, 22, 33
        for trial in range(TRIALS_PER_SUBJECT_PER_CLASS):
            metadata.append({
                'subject': subj_id,
                'Condition': condition,  # Expected by cardinality_1_3 label function
                'trial': trial
            })
    
    import pandas as pd
    metadata_df = pd.DataFrame(metadata)
    
    # Create Epochs object
    events = np.column_stack([
        np.arange(n_epochs) * 200,  # onset samples
        np.zeros(n_epochs, dtype=int),  # duration
        np.repeat([1, 2, 3], TRIALS_PER_SUBJECT_PER_CLASS)  # event_id
    ])
    
    epochs = mne.EpochsArray(
        data=data,
        info=info,
        tmin=times[0],
        metadata=metadata_df,
        events=events,
        verbose=False
    )
    
    # Save .fif file
    fif_path = DATA_DIR / f"sub-{subj_id:02d}_preprocessed-epo.fif"
    epochs.save(fif_path, overwrite=True, verbose=False)
    print(f"  -> Saved {fif_path.name} ({n_epochs} epochs)")

# Create minimal model checkpoints (2 folds)
print("\nCreating mock checkpoints...")

# Need to create actual EEGNeX model checkpoints
from code.model_builders import build_eegnex

# Create a minimal EEGNeX model and save its state dict
cfg = {
    'activation': 'elu',
    'depth_multiplier': 2,
    'filter_1': 8,
    'filter_2': 32,
    'drop_prob': 0.5,
}

for fold in [1, 2]:
    # Build model with same config as fixture
    model = build_eegnex(cfg, num_classes=N_CLASSES, C=N_CHANNELS, T=N_TIMES)
    state_dict = model.state_dict()
    
    ckpt_path = CKPT_DIR / f"fold_{fold:02d}_best.ckpt"
    torch.save(state_dict, ckpt_path)
    print(f"  -> Saved {ckpt_path.name}")

# Create summary.json
print("\nCreating summary.json...")

summary = {
    "run_id": "test_xai_fixture_001",
    "dataset_dir": str(DATA_DIR.absolute()),
    "hyper": {
        "task": "cardinality_1_3",
        "model_name": "eegnex",  # lowercase per RAW_EEG_MODELS
        "seed": 42,
        "materialized_dir": str(DATA_DIR.absolute()),
        "xai_top_k_channels": 10,
        "peak_window_ms": 100,
        "tf_morlet_freqs": [4, 8, 13, 30],
        "gradcam_target_layer": "features.3"
    },
    "fold_splits": [
        {
            "fold": 1,
            "test_subjects": [1],
            "train_subjects": [2]
        },
        {
            "fold": 2,
            "test_subjects": [2],
            "train_subjects": [1]
        }
    ]
}

summary_path = FIXTURE_DIR / "summary_test_xai_fixture_001.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"  -> Saved {summary_path.name}")
print("\nTest fixture created successfully!")
print(f"  Location: {FIXTURE_DIR}")
print(f"  Subjects: 2")
print(f"  Classes: 3 (cardinality 1, 2, 3)")
print(f"  Trials per subject per class: {TRIALS_PER_SUBJECT_PER_CLASS}")
print(f"  Channels: {N_CHANNELS}")
print(f"  Time points: {N_TIMES}")
print(f"  Sampling rate: {SFREQ} Hz")

