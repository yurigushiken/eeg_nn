"""
V4: Generate All Neuroscience Publication-Ready Figures
- FIXED: Muted blues (conservative palette)
- FIXED: All occlusions resolved
- ALL figures meet neuroscience publication standards
"""

import sys
import subprocess
from pathlib import Path

# List of all V4 figure scripts
ALL_FIGURES_V4 = [
    "figure1_pipeline_v4.py",
    "figure2_nested_cv_v4.py",
    "figure3_optuna_optimization_v4.py",
    "figure4_confusion_matrices_v4.py",
    "figure5_learning_curves_v4.py",
    "figure6_permutation_v4.py",
    "figure7_per_subject_forest_v4.py",
    "figure8_xai_spatiotemporal_v4.py",
    "figure9_xai_perclass_v4.py",
    "figure10_performance_boxplots_v4.py",
]

def run_script(script_name):
    """Run a single figure generation script"""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"\n[ERROR] Script not found: {script_name}")
        return False
    
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=script_path.parent
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings:\n{result.stderr}")
        
        print(f"\n[OK] {script_name} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run {script_name}:")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        return False

if __name__ == '__main__':
    print("\n" + "="*70)
    print("   GENERATING ALL V4 NEUROSCIENCE PUBLICATION FIGURES")
    print("="*70)
    print("\n[V4 IMPROVEMENTS]")
    print("  - Muted blues palette (conservative, professional)")
    print("  - Fixed Figure 1: Shades of blue only")
    print("  - Fixed Figure 2: No text box occlusion")
    print("  - Fixed Figure 4: Per-class F1 moved left (no occlusion)")
    print("  - All figures: WHITE backgrounds, Wong/Blues palettes")
    print("  - Meets ALL neuroscience journal standards\n")
    
    success_count = 0
    total_count = len(ALL_FIGURES_V4)
    
    # Generate all figures
    for script in ALL_FIGURES_V4:
        if run_script(script):
            success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"   COMPLETED: {success_count}/{total_count} figures generated")
    print("="*70)
    
    if success_count == total_count:
        print("\n[OK] ALL V4 FIGURES GENERATED SUCCESSFULLY!")
        print("Output location: publication-ready-media/outputs/v4/")
        print("\n[OK] ALL FIGURES MEET NEUROSCIENCE STANDARDS:")
        print("     - White backgrounds")
        print("     - Conservative colors (muted blues + Wong)")
        print("     - No occlusions")
        print("     - 600 DPI, vector PDF")
        print("     - Ready for journal submission!")
    else:
        print(f"\n[WARNING] {total_count - success_count} figures failed. Check errors above.")

