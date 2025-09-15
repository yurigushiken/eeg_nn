from .eeg import run as eeg_run

ENGINES = {
    "eeg": eeg_run,
}

def get(name: str):
    if name not in ENGINES:
        raise KeyError(f"Unknown engine '{name}'. Registered: {list(ENGINES.keys())}")
    return ENGINES[name]


