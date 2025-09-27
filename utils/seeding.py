import os
import random
import numpy as np
import torch


def seed_everything(seed: int | None):
    if seed is None:
        return
    # Environment seeds
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Ensure deterministic GEMM paths in CUDA where applicable
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # alternative: ":16:8" for small workspace

    # Python/NumPy/Torch seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN + PyTorch deterministic mode
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Fail loudly if a non-deterministic op is used
    torch.use_deterministic_algorithms(True, warn_only=False)


def determinism_banner(seed: int | None) -> dict:
    """Return a dict describing active determinism settings for auditing."""
    try:
        cudnn_det = torch.backends.cudnn.deterministic
        cudnn_bench = torch.backends.cudnn.benchmark
    except Exception:
        cudnn_det = None
        cudnn_bench = None
    return {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn.benchmark": cudnn_bench,
        "cudnn.deterministic": cudnn_det,
        "torch.use_deterministic_algorithms": True,
    }


