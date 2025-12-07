"""
Test-driven development for number sense feature map symmetry.

This test ensures the plot has visual symmetry with balanced whitespace
on left and right sides to compensate for y-axis label text.
"""
import pytest
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing


def test_feature_map_has_symmetric_margins():
    """
    Test that the feature map plot has symmetric left/right margins.

    The y-axis labels take up visual space on the left, so we need
    equivalent whitespace on the right for visual balance.

    This test should FAIL initially, then PASS after fixing the plot.
    """
    # Import the script as a module
    import sys
    from pathlib import Path

    # Add number_sense directory to path
    number_sense_dir = Path(__file__).parent.parent / "number_sense"
    sys.path.insert(0, str(number_sense_dir))

    # Import and run the script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_feature_map",
        number_sense_dir / "generate_feature_map.py"
    )
    module = importlib.util.module_from_spec(spec)

    # Execute the script to generate the plot
    spec.loader.exec_module(module)

    # Get the current figure
    fig = plt.gcf()

    # Get subplot parameters
    left = fig.subplotpars.left
    right = fig.subplotpars.right

    # Calculate actual margins (fraction of figure width)
    left_margin = left
    right_margin = 1.0 - right

    print(f"\n[Symmetry Test]")
    print(f"  Left margin:  {left_margin:.3f} ({left_margin*100:.1f}% of figure width)")
    print(f"  Right margin: {right_margin:.3f} ({right_margin*100:.1f}% of figure width)")
    print(f"  Difference:   {abs(left_margin - right_margin):.3f}")

    # Test: margins should be symmetric within 1% tolerance
    # This allows for minor differences while ensuring visual balance
    assert abs(left_margin - right_margin) < 0.01, (
        f"Left and right margins are not symmetric!\n"
        f"  Left margin:  {left_margin:.3f}\n"
        f"  Right margin: {right_margin:.3f}\n"
        f"  Difference:   {abs(left_margin - right_margin):.3f}\n"
        f"Expected symmetric margins (difference < 0.01)"
    )

    # Additional check: both margins should be substantial (> 15%)
    # This ensures we have actual whitespace, not just equal zeros
    assert left_margin > 0.15, (
        f"Left margin too small: {left_margin:.3f} (< 0.15)"
    )
    assert right_margin > 0.15, (
        f"Right margin too small: {right_margin:.3f} (< 0.15)"
    )

    print(f"  [OK] Margins are symmetric and substantial!")

    # Clean up
    plt.close('all')


def test_feature_map_plot_box_is_centered():
    """
    Test that the actual plot area (data region) is centered in the figure.

    This is a secondary check to ensure the plot box itself is visually centered.
    """
    import sys
    from pathlib import Path

    number_sense_dir = Path(__file__).parent.parent / "number_sense"
    sys.path.insert(0, str(number_sense_dir))

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_feature_map",
        number_sense_dir / "generate_feature_map.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fig = plt.gcf()

    left = fig.subplotpars.left
    right = fig.subplotpars.right

    # Calculate plot box width and center
    plot_width = right - left
    plot_center = left + plot_width / 2.0
    figure_center = 0.5

    print(f"\n[Centering Test]")
    print(f"  Plot box spans: {left:.3f} to {right:.3f}")
    print(f"  Plot box width: {plot_width:.3f}")
    print(f"  Plot center:    {plot_center:.3f}")
    print(f"  Figure center:  {figure_center:.3f}")
    print(f"  Offset:         {abs(plot_center - figure_center):.3f}")

    # Plot box should be centered within 1% of figure center
    assert abs(plot_center - figure_center) < 0.01, (
        f"Plot box is not centered!\n"
        f"  Plot center:   {plot_center:.3f}\n"
        f"  Figure center: {figure_center:.3f}\n"
        f"  Offset:        {abs(plot_center - figure_center):.3f}\n"
        f"Expected centered plot (offset < 0.01)"
    )

    print(f"  [OK] Plot box is centered!")

    plt.close('all')


if __name__ == "__main__":
    # Run tests manually for debugging
    print("=" * 60)
    print("Running Feature Map Symmetry Tests")
    print("=" * 60)

    try:
        test_feature_map_has_symmetric_margins()
        print("\n[PASS] Test 1 PASSED: Symmetric margins")
    except AssertionError as e:
        print(f"\n[FAIL] Test 1 FAILED: {e}")

    try:
        test_feature_map_plot_box_is_centered()
        print("\n[PASS] Test 2 PASSED: Centered plot box")
    except AssertionError as e:
        print(f"\n[FAIL] Test 2 FAILED: {e}")

    print("\n" + "=" * 60)
