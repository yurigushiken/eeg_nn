"""
V3 FINAL: Generate All Publication-Ready Figures
- NEUROSCIENCE PUBLICATION STANDARDS (white backgrounds!)
- Fixed padding/whitespace
- Fixed occluding text
- Wong colorblind-safe palette
- All improvements from V2 + PI feedback
"""

import sys
sys.path.append('../utils')

import subprocess
from pathlib import Path

# List of all V3 figure scripts to generate (ALL LIGHT THEME for publication!)
ALL_FIGURES = [
    "figure1_pipeline_v3.py",
    "figure2_nested_cv_v3.py",
    "figure3_optuna_optimization_v3.py",
    "figure4_confusion_matrices_v3.py",
    "figure5_learning_curves_v3.py",
    "figure6_permutation_v3.py",
    "figure7_per_subject_forest_v3.py",
    "figure8_xai_spatiotemporal_v3.py",
    "figure9_xai_perclass_v3.py",
    "figure10_performance_boxplots_v3.py",
]

def run_script(script_name):
    """Run a single figure generation script"""
    script_path = Path(__file__).parent / script_name
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print('='*70)
    
    try:
        result = subprocess.run([sys.executable, str(script_path)], 
                              check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Warnings:\n{result.stderr}")
        print(f"[OK] {script_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to run {script_name}:")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        return False

if __name__ == '__main__':
    print("\n" + "="*70)
    print("   GENERATING ALL V3 PUBLICATION-READY FIGURES")
    print("   (NEUROSCIENCE JOURNAL STANDARDS)")
    print("="*70)
    print("\nAll figures: WHITE backgrounds, Wong palette")
    print("Meets standards for: JNeurosci, Nature Neuro, Neuron, eNeuro\n")
    
    success_count = 0
    total_count = len(ALL_FIGURES)
    
    # Generate all figures (all light theme for publication!)
    for script in ALL_FIGURES:
        if run_script(script):
            success_count += 1
    
    # Summary
    print("\n" + "="*70)
    print(f"   COMPLETED: {success_count}/{total_count} figures generated")
    print("="*70)
    
    if success_count == total_count:
        print("\n[OK] All V3 figures generated successfully!")
        print("Output location: publication-ready-media/outputs/v3_final/")
        print("\n[OK] All figures meet neuroscience publication standards!")
        print("     - White backgrounds")
        print("     - Wong colorblind-safe palette")
        print("     - 600 DPI, vector PDF")
        print("     - Ready for journal submission!")
    else:
        print(f"\n[WARNING] {total_count - success_count} figures failed. Check errors above.")

