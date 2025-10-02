from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import pytest

import mne


# ---------------------------------------------------------------------------
# Deterministic defaults for synthetic EEG epochs used in tests.
# These fixtures provide lightweight, materialized .fif files so that tests can
# exercise subject metadata, channel alignment, and 5-trial thresholds without
# relying on production datasets.
# ---------------------------------------------------------------------------

TEST_SFREQ = 128.0
TEST_DURATION_S = 0.7
TEST_TMIN = -0.1
TEST_SEED = 1337

DEFAULT_CHANNELS = ("Fz", "Cz", "Pz", "Oz")
DEFAULT_CLASS_COUNTS = {"num_1": 6, "num_2": 6}


@dataclass(frozen=True)
class SubjectSpec:
    """Specification for a synthetic subject used in fixtures.

    Attributes
    ----------
    subject_id: int
        Numeric identifier used in filenames (`sub-<id>` convention).
    class_counts: Mapping[str, int]
        Mapping of class label → number of trials.
    channels: Tuple[str, ...]
        EEG channel names materialized for the subject.
    """

    subject_id: int
    class_counts: Mapping[str, int]
    channels: Tuple[str, ...] = DEFAULT_CHANNELS

    @property
    def total_trials(self) -> int:
        return int(sum(int(v) for v in self.class_counts.values()))


@dataclass(frozen=True)
class SyntheticMaterializedDataset:
    """Container describing a synthetic materialized dataset."""

    root: Path
    specs: Tuple[SubjectSpec, ...]
    condition_codes: Dict[str, int]
    trial_metadata: pd.DataFrame
    channels_by_subject: Dict[int, Tuple[str, ...]]
    sfreq: float
    times_ms: np.ndarray

    @property
    def materialized_dir(self) -> Path:
        return self.root


def _derive_condition_codes(specs: Sequence[SubjectSpec]) -> Dict[str, int]:
    labels: Iterable[str] = {
        label
        for spec in specs
        for label in spec.class_counts.keys()
    }
    return {label: idx for idx, label in enumerate(sorted(labels), start=100)}


def _build_epochs(spec: SubjectSpec, condition_codes: Mapping[str, int]) -> Tuple[mne.EpochsArray, pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(TEST_SEED + spec.subject_id)
    n_times = int(TEST_SFREQ * TEST_DURATION_S)
    info = mne.create_info(
        ch_names=list(spec.channels),
        sfreq=TEST_SFREQ,
        ch_types="eeg",
    )

    data = []
    rows = []
    trend = np.linspace(-0.5, 0.5, n_times, dtype=np.float32)
    trial_idx = 0
    for label in sorted(spec.class_counts.keys()):
        count = int(spec.class_counts[label])
        code = condition_codes[label]
        for local_idx in range(count):
            noise = rng.normal(scale=1e-6, size=(len(spec.channels), n_times))
            signal = noise + trend * (code / 100.0)
            data.append(signal.astype(np.float32))
            rows.append(
                {
                    "subject": spec.subject_id,
                    "subject_str": f"sub-{spec.subject_id:02d}",
                    "trial_index": trial_idx,
                    "label": label,
                    "condition_code": code,
                    "quality_flag": "good",
                    "event_time_ms": float((TEST_TMIN + local_idx / TEST_SFREQ) * 1000.0),
                }
            )
            trial_idx += 1

    epochs = mne.EpochsArray(
        np.asarray(data, dtype=np.float32),
        info=info,
        tmin=TEST_TMIN,
        verbose=False,
    )
    metadata = pd.DataFrame(rows)
    epochs.metadata = metadata
    times_ms = (epochs.times * 1000.0).astype(np.float32)
    return epochs, metadata, times_ms


@pytest.fixture(scope="session", autouse=True)
def _set_mne_log_level() -> None:
    """Keep MNE quiet during test runs."""

    mne.set_log_level("WARNING")


@pytest.fixture(scope="session")
def synthetic_subject_specs() -> Tuple[SubjectSpec, ...]:
    """Default set of subjects covering ≥5-trial and <5-trial scenarios."""

    return (
        SubjectSpec(subject_id=1, class_counts=DEFAULT_CLASS_COUNTS),
        SubjectSpec(subject_id=2, class_counts=DEFAULT_CLASS_COUNTS),
        SubjectSpec(subject_id=4, class_counts=DEFAULT_CLASS_COUNTS),
        SubjectSpec(subject_id=5, class_counts=DEFAULT_CLASS_COUNTS),
        # Include one subject below threshold to exercise exclusion logic
        SubjectSpec(
            subject_id=3,
            class_counts={"num_1": 4, "num_2": 4},
            channels=("Fz", "Cz", "Oz"),
        ),
    )


@pytest.fixture
def materialized_dataset(tmp_path_factory: pytest.TempPathFactory, synthetic_subject_specs: Tuple[SubjectSpec, ...]) -> SyntheticMaterializedDataset:
    """Create a deterministic, materialized dataset for tests.

    Returns a `SyntheticMaterializedDataset` describing the generated `.fif`
    files so tests can reference metadata, channel sets, or reuse the directory
    when instantiating `MaterializedEpochsDataset` from the production code.
    """

    dataset_dir = tmp_path_factory.mktemp("materialized_epochs")
    specs = synthetic_subject_specs
    condition_codes = _derive_condition_codes(specs)

    all_metadata: list[pd.DataFrame] = []
    times_template: np.ndarray | None = None
    channels_by_subject: Dict[int, Tuple[str, ...]] = {}

    for spec in specs:
        epochs, metadata, times_ms = _build_epochs(spec, condition_codes)
        fname = dataset_dir / f"sub-{spec.subject_id:02d}_preprocessed-epo.fif"
        epochs.save(fname, overwrite=True)
        all_metadata.append(metadata.assign(file_path=str(fname)))
        channels_by_subject[spec.subject_id] = spec.channels
        times_template = times_ms if times_template is None else times_template

    trial_metadata = pd.concat(all_metadata, ignore_index=True)

    return SyntheticMaterializedDataset(
        root=dataset_dir,
        specs=specs,
        condition_codes=condition_codes,
        trial_metadata=trial_metadata,
        channels_by_subject=channels_by_subject,
        sfreq=TEST_SFREQ,
        times_ms=times_template if times_template is not None else np.array([], dtype=np.float32),
    )


@pytest.fixture
def subject_trial_counts(materialized_dataset: SyntheticMaterializedDataset) -> pd.DataFrame:
    """Aggregate trial counts per subject/class from the synthetic dataset."""

    counts = (
        materialized_dataset.trial_metadata.groupby(["subject", "label"])  # type: ignore[arg-type]
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    counts.attrs["condition_codes"] = materialized_dataset.condition_codes
    return counts


@pytest.fixture
def materialized_dir(materialized_dataset: SyntheticMaterializedDataset) -> Path:
    """Convenience fixture exposing the path to synthetic `.fif` files."""

    return materialized_dataset.materialized_dir

