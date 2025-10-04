"""
XAI utilities for EEG models.

Provides:
- Integrated Gradients (IG) computation and visualization
- Heatmap and topomap plotting
- Time-frequency analysis

All functions are designed to work with EEG models and MNE-Python for visualization.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import mne

try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None


# ==================== T005: Integrated Gradients ====================

def compute_ig_attributions(
    model: nn.Module,
    dataset: Dataset,
    test_idx: List[int],
    device: torch.device,
    class_names: Optional[List[str]] = None,
    input_adapter=None
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Compute Integrated Gradients attributions for correctly classified test samples.
    
    Args:
        model: PyTorch model to explain
        dataset: Dataset containing EEG data (expects .X as (N, 1, C, T))
        test_idx: Indices of test samples
        device: torch.device for computation
        class_names: Optional list of class names
        input_adapter: Optional function to adapt input shape (e.g., squeeze)
        
    Returns:
        Tuple of:
        - attr_matrix: (C, T) numpy array of averaged attributions
        - n_correct: Number of correctly classified samples
        - trial_labels: (n_correct,) array of class labels for each trial
    """
    if IntegratedGradients is None:
        raise ImportError("Captum not available. Install with: pip install captum")
    
    model.eval()
    model.to(device)
    
    # Prepare data
    X_test = dataset.X[test_idx].to(device)  # (N, 1, C, T)
    y_test = dataset.y[test_idx].to(device)  # (N,)
    
    # Apply input adapter if provided
    X_test_adapted = input_adapter(X_test) if input_adapter else X_test
    
    # Get predictions to filter correctly classified samples
    with torch.no_grad():
        outputs = model(X_test_adapted)
        predictions = outputs.argmax(dim=1)
        correct_mask = (predictions == y_test).cpu().numpy()
    
    # Filter to correctly classified samples only
    correct_indices = np.where(correct_mask)[0]
    n_correct = len(correct_indices)
    
    if n_correct == 0:
        print("  WARNING: No correctly classified samples in this fold")
        # Return zeros
        C, T = X_test.shape[2], X_test.shape[3]
        return np.zeros((C, T), dtype=np.float32), 0, np.array([])
    
    X_correct = X_test[correct_indices]  # (n_correct, 1, C, T)
    y_correct = y_test[correct_indices]  # (n_correct,)
    trial_labels = y_correct.cpu().numpy()
    
    # If input_adapter is provided, wrap model to include it
    if input_adapter:
        class AdaptedModel(nn.Module):
            def __init__(self, base_model, adapter):
                super().__init__()
                self.model = base_model
                self.adapter = adapter
            
            def forward(self, x):
                return self.model(self.adapter(x))
        
        model_for_ig = AdaptedModel(model, input_adapter)
    else:
        model_for_ig = model
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(model_for_ig)
    
    # Compute attributions for each correct sample
    attributions_list = []
    
    for i in range(n_correct):
        x_sample = X_correct[i:i+1]  # (1, 1, C, T)
        target = y_correct[i:i+1]    # (1,)
        
        # Compute IG attributions
        attr = ig.attribute(
            x_sample,
            target=target,
            n_steps=50,  # Number of integration steps
            internal_batch_size=1
        )
        
        # attr shape: (1, 1, C, T) - squeeze batch & channel dims to (C, T)
        # We need to handle both 4D (1,1,C,T) and potentially 3D (1,C,T) outputs
        attr_np = attr.squeeze(0).cpu().numpy()  # Remove batch dim
        if attr_np.ndim == 3:  # (1, C, T) - remove leading 1
            attr_np = attr_np.squeeze(0)
        attributions_list.append(attr_np)  # (C, T)
    
    # Average across all correctly classified samples
    attr_matrix = np.mean(attributions_list, axis=0)  # (C, T)
    
    print(f"  -> Computed IG for {n_correct}/{len(test_idx)} correctly classified samples")
    
    return attr_matrix, n_correct, trial_labels


# ==================== T007: Attribution Heatmap ====================

def plot_attribution_heatmap(
    attr_matrix: np.ndarray,
    ch_names: List[str],
    times_ms: np.ndarray,
    title: str,
    output_path: Path
) -> None:
    """
    Plot and save an attribution heatmap (channels × time).
    
    Args:
        attr_matrix: (C, T) attribution matrix
        ch_names: List of channel names (length C)
        times_ms: Time points in milliseconds (length T)
        title: Plot title
        output_path: Path to save PNG
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap
    im = ax.imshow(
        attr_matrix,
        cmap='inferno',
        aspect='auto',
        interpolation='nearest'
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('EEG Channels', fontsize=12)
    
    # Set y-axis (channels)
    ax.set_yticks(np.arange(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=8)
    
    # Set x-axis (time)
    n_time_ticks = min(10, len(times_ms))
    xt = np.linspace(0, len(times_ms) - 1, num=n_time_ticks, dtype=int)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{times_ms[i]:.0f}" for i in xt], fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Attribution Intensity')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  -> Saved heatmap: {output_path.name}")


# ==================== T008: Topomap ====================

def plot_topomap(
    channel_importance: np.ndarray,
    info: mne.Info,
    top_k_indices: np.ndarray,
    ch_names: List[str],
    title: str,
    output_path: Path
) -> None:
    """
    Plot and save a topomap with top-K channels labeled.
    
    Args:
        channel_importance: (C,) array of channel importance values
        info: MNE Info object with montage attached
        top_k_indices: Indices of top-K channels to label
        ch_names: List of all channel names
        title: Plot title
        output_path: Path to save PNG
    """
    # Compute robust color limits
    v = np.abs(channel_importance[np.isfinite(channel_importance)])
    if v.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(np.percentile(v, 2.0))
        vmax = float(np.percentile(v, 98.0))
        if vmax <= 0 or vmax <= vmin:
            vmax = float(np.max(v)) if np.max(v) > 0 else 1.0
            vmin = 0.0
    
    # Create labels (only top-K channels)
    names = [""] * len(ch_names)
    for idx in top_k_indices:
        names[idx] = ch_names[idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot topomap
    im, cn = mne.viz.plot_topomap(
        channel_importance,
        info,
        axes=ax,
        show=False,
        cmap='inferno',
        names=names,
        vlim=(vmin, vmax),
        contours=8
    )
    
    # Adjust contour line width for better visibility
    if cn is not None:
        artists = getattr(cn, "collections", cn if isinstance(cn, (list, tuple)) else [])
        for artist in artists:
            try:
                artist.set_linewidth(1.0)
            except Exception:
                pass
    
    # Set title
    ax.set_title(title, fontsize=14, pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Attribution Intensity", rotation=90, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    fig.savefig(output_path, bbox_inches='tight', dpi=180)
    plt.close(fig)
    
    print(f"  -> Saved topomap: {output_path.name}")


# ==================== T009: Time-Frequency Computation ====================

def compute_time_frequency_map(
    attr_matrix: np.ndarray,
    sfreq: float,
    freqs: List[float]
) -> Optional[Any]:
    """
    Compute time-frequency representation using Morlet wavelets.
    
    Args:
        attr_matrix: (C, T) attribution matrix to analyze
        sfreq: Sampling frequency in Hz
        freqs: List of frequencies for TFR decomposition
        
    Returns:
        TFR object from MNE, or None if signal too short for wavelets
    """
    # Create a synthetic MNE Epochs object from attribution matrix
    # Reshape to (n_epochs=1, n_channels, n_times)
    n_channels, n_times = attr_matrix.shape
    
    # Check if signal is long enough for the lowest frequency
    # Wavelet length ≈ n_cycles * sfreq / freq
    # Use minimum n_cycles of 2.0
    min_n_cycles = 2.0
    min_freq = min(freqs)
    required_samples = int(min_n_cycles * sfreq / min_freq)
    
    if n_times < required_samples:
        print(f"  -> Signal too short ({n_times} samples) for TFR with lowest freq {min_freq} Hz")
        print(f"     (requires >= {required_samples} samples). Skipping time-frequency analysis.")
        return None
    
    data = attr_matrix[np.newaxis, :, :]  # (1, C, T)
    
    # Create MNE Info object
    ch_names = [f"ch_{i+1}" for i in range(n_channels)]
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=['eeg'] * n_channels
    )
    
    # Create synthetic events (required by EpochsArray)
    events = np.array([[0, 0, 1]])  # Single event at time 0
    
    # Create EpochsArray
    epochs = mne.EpochsArray(
        data=data,
        info=info,
        events=events,
        tmin=0.0,
        verbose=False
    )
    
    # Compute time-frequency representation using Morlet wavelets
    n_cycles = np.array(freqs) / 2.0  # Number of cycles increases with frequency
    
    try:
        tfr = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=True,
            verbose=False
        )
        print(f"  -> Computed TFR with {len(freqs)} frequency bins")
        return tfr
    except ValueError as e:
        print(f"  -> TFR computation failed: {e}")
        return None


# ==================== T010: Time-Frequency Plot ====================

def plot_time_frequency_map(
    tfr: Any,
    output_path: Path
) -> None:
    """
    Plot and save a time-frequency heatmap.
    
    Args:
        tfr: MNE TFR object (AverageTFR)
        output_path: Path to save PNG
    """
    # Average across channels for visualization
    # tfr.data shape: (n_channels, n_freqs, n_times)
    tfr_avg = np.mean(np.abs(tfr.data), axis=0)  # (n_freqs, n_times)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time-frequency heatmap
    extent = [
        tfr.times[0] * 1000,  # Convert to ms
        tfr.times[-1] * 1000,
        tfr.freqs[0],
        tfr.freqs[-1]
    ]
    
    im = ax.imshow(
        tfr_avg,
        aspect='auto',
        origin='lower',
        cmap='inferno',
        extent=extent,
        interpolation='bilinear'
    )
    
    # Labels and title
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title('Time-Frequency Attribution (Grand Average)', fontsize=14)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Power')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  -> Saved time-frequency plot: {output_path.name}")


# ==================== Legacy compatibility stubs ====================

def compute_and_plot_attributions(*args, **kwargs):
    """Legacy stub - use individual functions instead."""
    return None
