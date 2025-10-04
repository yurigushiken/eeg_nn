"""
Master Script: Generate All 10 Publication-Ready Figures

Run this to regenerate all figures at once.
Figures will be saved to ../../outputs/v2_improved/
"""

import sys
import subprocess
from pathlib import Path

# List of all figure scripts in order
figure_scripts = [
    "figure1_pipeline_v2.py",
    "figure2_nested_cv_v2.py",
    "figure3_optuna_optimization.py",
    "figure4_confusion_matrices.py",
    "figure5_learning_curves.py",
    "figure6_permutation_v2.py",
    "figure7_per_subject_forest.py",
    "figure8_xai_spatiotemporal.py",
    "figure9_xai_perclass.py",
    "figure10_performance_boxplots.py",
]

def main():
    """Run all figure generation scripts"""
    print("\n" + "="*70)
    print("GENERATING ALL 10 PUBLICATION-READY FIGURES")
    print("="*70 + "\n")
    
    success_count = 0
    failed = []
    
    for i, script in enumerate(figure_scripts, 1):
        print(f"\n[{i}/10] Running {script}...")
        print("-" * 70)
        
        try:
            result = subprocess.run(
                [sys.executable, script],
                capture_output=False,
                text=True,
                check=True
            )
            success_count += 1
            print(f"✓ {script} completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ {script} FAILED")
            failed.append(script)
        except Exception as e:
            print(f"✗ {script} ERROR: {e}")
            failed.append(script)
        
        print("-" * 70)
    
    # Summary
    print("\n" + "="*70)
    print(f"SUMMARY: {success_count}/10 figures generated successfully")
    print("="*70)
    
    if failed:
        print("\n❌ Failed scripts:")
        for script in failed:
            print(f"   - {script}")
        print("\nPlease run failed scripts individually to see error details.")
    else:
        print("\n✅ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("\nOutputs saved to: ../../outputs/v2_improved/")
        print("  - PDF (vector) - for publications")
        print("  - PNG (600 DPI) - for PowerPoint/posters")
        print("  - SVG (vector) - for editing")

if __name__ == '__main__':
    main()

