"""
Contract test for XAI configuration schema.

Validates that configs/xai_defaults.yaml contains all required fields
with correct types, ensuring the XAI system can load configuration properly.
"""

import pytest
import yaml
from pathlib import Path


@pytest.fixture
def xai_config_path():
    """Return path to xai_defaults.yaml."""
    return Path(__file__).parent.parent.parent / "configs" / "xai_defaults.yaml"


@pytest.fixture
def xai_config(xai_config_path):
    """Load and return xai_defaults.yaml content."""
    assert xai_config_path.exists(), f"xai_defaults.yaml not found at {xai_config_path}"
    with open(xai_config_path, 'r') as f:
        return yaml.safe_load(f)


def test_xai_config_exists(xai_config_path):
    """Test that xai_defaults.yaml file exists."""
    assert xai_config_path.exists(), "xai_defaults.yaml file must exist"


def test_xai_config_has_peak_window_ms(xai_config):
    """Test that peak_window_ms is present and is a number."""
    assert 'peak_window_ms' in xai_config, "peak_window_ms must be present"
    assert isinstance(xai_config['peak_window_ms'], (int, float)), \
        "peak_window_ms must be a number"
    assert xai_config['peak_window_ms'] > 0, "peak_window_ms must be positive"


def test_xai_config_has_xai_top_k_channels(xai_config):
    """Test that xai_top_k_channels is present and is a positive integer."""
    assert 'xai_top_k_channels' in xai_config, "xai_top_k_channels must be present"
    assert isinstance(xai_config['xai_top_k_channels'], int), \
        "xai_top_k_channels must be an integer"
    assert xai_config['xai_top_k_channels'] > 0, "xai_top_k_channels must be positive"


def test_xai_config_has_tf_morlet_freqs(xai_config):
    """Test that tf_morlet_freqs is present and is a list of numbers."""
    assert 'tf_morlet_freqs' in xai_config, "tf_morlet_freqs must be present"
    assert isinstance(xai_config['tf_morlet_freqs'], list), \
        "tf_morlet_freqs must be a list"
    assert len(xai_config['tf_morlet_freqs']) > 0, \
        "tf_morlet_freqs must not be empty"
    assert all(isinstance(f, (int, float)) for f in xai_config['tf_morlet_freqs']), \
        "All frequencies in tf_morlet_freqs must be numbers"
    assert all(f > 0 for f in xai_config['tf_morlet_freqs']), \
        "All frequencies must be positive"


def test_xai_config_has_gradcam_target_layer(xai_config):
    """Test that gradcam_target_layer is present (can be string or null)."""
    assert 'gradcam_target_layer' in xai_config, "gradcam_target_layer must be present"
    # Can be either a string (layer name) or null
    assert xai_config['gradcam_target_layer'] is None or \
           isinstance(xai_config['gradcam_target_layer'], str), \
        "gradcam_target_layer must be either a string or null"


def test_xai_config_schema_complete(xai_config):
    """Test that all required fields are present."""
    required_fields = {
        'peak_window_ms',
        'xai_top_k_channels',
        'tf_morlet_freqs',
        'gradcam_target_layer'
    }
    
    present_fields = set(xai_config.keys())
    missing_fields = required_fields - present_fields
    
    assert not missing_fields, \
        f"Missing required fields in xai_defaults.yaml: {missing_fields}"


def test_xai_config_realistic_values(xai_config):
    """Test that configuration values are within reasonable ranges."""
    # Peak window should be reasonable (10-500 ms typical for EEG)
    assert 10 <= xai_config['peak_window_ms'] <= 500, \
        "peak_window_ms should be between 10 and 500 ms"
    
    # Top-K channels should be reasonable (1-129 for 128-channel systems)
    assert 1 <= xai_config['xai_top_k_channels'] <= 129, \
        "xai_top_k_channels should be between 1 and 129"
    
    # Frequency bands should be in EEG range (typically 1-100 Hz)
    assert all(1 <= f <= 100 for f in xai_config['tf_morlet_freqs']), \
        "All frequencies should be in typical EEG range (1-100 Hz)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

