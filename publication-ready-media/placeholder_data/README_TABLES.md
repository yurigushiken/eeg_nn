# Publication Tables - Usage Guide

**Purpose:** This directory contains CSV templates for all publication tables. Use these as templates to structure your real data.

---

## 📊 **AVAILABLE TABLES**

### ✅ **Created (with placeholder data)**

1. **`table1_demographics.csv`**
   - Participant demographics and data quality metrics
   - 24 rows (subjects) × 11 columns
   - Use for: Methods section, participant characterization

2. **`table2_hyperparameters.csv`**
   - Hyperparameter search space and best values
   - 18 rows (parameters) × 7 columns
   - Use for: Methods section, HPO documentation

3. **`table4_performance_aggregated.csv`**
   - Aggregated performance comparison across tasks
   - 8 rows (metrics × tasks) × 9 columns
   - Use for: Results section, main performance table

4. **`table6_statistical_validation.csv`**
   - Permutation testing and statistical significance
   - 4 rows (metrics × tasks) × 10 columns
   - Use for: Results section, statistical validation

5. **`table9_computational_resources.csv`**
   - Computational requirements and reproducibility
   - 9 rows (processes) × 8 columns
   - Use for: Methods section, reproducibility statement

### 📝 **To Be Created (when you have real data)**

6. **`table3_performance_per_fold.csv`**
   - Per-fold performance metrics (24 folds × 2 tasks = 48 rows)
   - Will be LONG - consider supplementary materials
   - Source: `results/[study]/outer_eval_metrics.csv`

7. **`table5_confusion_matrices.csv`**
   - Aggregated confusion matrices
   - 6 rows (3 classes × 2 tasks) × 9 columns
   - Source: Sum across all fold confusion matrices

8. **`table7_xai_top_features.csv`**
   - Top contributing channels and time windows from XAI
   - ~30-50 rows (top features per class)
   - Source: XAI analysis output

9. **`table8_training_dynamics.csv`**
   - Learning curves summary and convergence metrics
   - 12 rows (metrics × tasks) × 6 columns
   - Source: `learning_curves.csv` aggregated

10. **`table10_per_subject_detailed.csv`**
    - Full per-subject breakdown (supplementary)
    - 24 rows (subjects) × 15+ columns
    - Source: Combine demographics + per-subject performance

---

## 🔄 **WORKFLOW: From Placeholder to Publication**

### Step 1: Collect Real Data

For each table, identify the source data:

```python
# Example: Table 3 (Performance per fold)
import pandas as pd
from pathlib import Path

study_dir = Path("../results/cardinality_1_3_study_final")
perf_csv = study_dir / "outer_eval_metrics.csv"

df = pd.read_csv(perf_csv)
# Process and format for table3_performance_per_fold.csv
```

### Step 2: Populate CSV Template

Replace placeholder values with real data, maintaining the same column structure:

```python
# Load template
template = pd.read_csv("table1_demographics.csv")

# Replace with real data (keep same columns!)
real_data = get_real_demographics()  # Your function
real_data.to_csv("table1_demographics.csv", index=False)
```

### Step 3: Generate LaTeX Tables

Use pandas or custom scripts to convert CSV → LaTeX:

```python
import pandas as pd

# Load data
df = pd.read_csv("table4_performance_aggregated.csv")

# Convert to LaTeX
latex_str = df.to_latex(
    index=False,
    escape=False,  # Allow LaTeX formatting
    column_format='lcccccccc',  # Alignment
    caption="Aggregated Performance Across Tasks",
    label="tab:performance_agg",
    float_format="%.2f"  # Decimal places
)

# Save
with open("table4_latex.tex", 'w') as f:
    f.write(latex_str)
```

### Step 4: Apply Journal-Specific Formatting

Each journal has specific table requirements:

**Nature/Science:**
- Use `\begin{table*}` for wide tables
- Caption above table
- Footnotes below table
- Times New Roman font

**PLOS:**
- Caption above table
- Upload as separate `.docx` file
- Include table title

**Cell:**
- Tables as editable files (Excel/.docx)
- Figure-quality tables as images

---

## 📋 **EXAMPLE: Complete Table Generation Script**

```python
"""
generate_publication_tables.py - Convert CSVs to LaTeX tables
"""

import pandas as pd
from pathlib import Path

def csv_to_latex(csv_path, caption, label, output_path=None):
    """Convert CSV to professional LaTeX table"""
    df = pd.read_csv(csv_path)
    
    # Determine alignment (left for text, center for numbers)
    col_format = ''
    for col in df.columns:
        if df[col].dtype == 'object':  # Text column
            col_format += 'l'
        else:  # Numeric column
            col_format += 'c'
    
    latex_str = df.to_latex(
        index=False,
        escape=False,
        column_format=col_format,
        caption=caption,
        label=label,
        float_format="%.2f",
        longtable=False,  # Set True for multi-page tables
        bold_rows=False
    )
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex_str)
        print(f"[OK] Saved: {output_path}")
    
    return latex_str

# Generate all tables
tables_config = [
    {
        "csv": "table1_demographics.csv",
        "caption": "Participant demographics and data quality metrics. " \
                  "All participants were right-handed unless noted. " \
                  "SNR = signal-to-noise ratio computed as mean across all channels.",
        "label": "tab:demographics",
        "output": "table1_demographics.tex"
    },
    {
        "csv": "table2_hyperparameters.csv",
        "caption": "Hyperparameter search space and best values selected by Optuna " \
                  "across three optimization stages. Best values shown for both " \
                  "Cardinality 1-3 and 4-6 tasks after 130 total trials.",
        "label": "tab:hyperparameters",
        "output": "table2_hyperparameters.tex"
    },
    {
        "csv": "table4_performance_aggregated.csv",
        "caption": "Aggregated performance metrics across tasks. " \
                  "Mean ± SD computed across 24 LOSO folds. " \
                  "P-values from permutation tests (200 permutations) vs. chance (33.3\\%). " \
                  "Cohen's d computed for accuracy metrics only.",
        "label": "tab:performance",
        "output": "table4_performance_aggregated.tex"
    },
    {
        "csv": "table6_statistical_validation.csv",
        "caption": "Statistical validation via permutation testing. " \
                  "Null distributions generated by shuffling labels within-subject " \
                  "while preserving class balance. CI = confidence interval (95\\%).",
        "label": "tab:statistical",
        "output": "table6_statistical_validation.tex"
    },
    {
        "csv": "table9_computational_resources.csv",
        "caption": "Computational resources and reproducibility details. " \
                  "All analyses were conducted with fixed random seeds for reproducibility. " \
                  "GPU: NVIDIA RTX 3090 (24GB VRAM).",
        "label": "tab:resources",
        "output": "table9_computational_resources.tex"
    }
]

if __name__ == '__main__':
    print("\n" + "="*70)
    print("   GENERATING PUBLICATION TABLES (LaTeX)")
    print("="*70 + "\n")
    
    output_dir = Path("latex_tables")
    output_dir.mkdir(exist_ok=True)
    
    for config in tables_config:
        csv_path = Path(config["csv"])
        output_path = output_dir / config["output"]
        
        print(f"\nProcessing: {csv_path.name}")
        latex_str = csv_to_latex(
            csv_path=csv_path,
            caption=config["caption"],
            label=config["label"],
            output_path=output_path
        )
    
    print("\n" + "="*70)
    print(f"   COMPLETE: All LaTeX tables in '{output_dir}/'")
    print("="*70 + "\n")
```

---

## 💡 **TIPS & BEST PRACTICES**

### Column Names
- Use underscores for spaces: `Mean_Trial_SNR_dB`
- LaTeX will convert underscores to subscripts
- Use clear, descriptive names

### Numeric Formatting
- Percentages: Include `%` in column name (`Accuracy_%`)
- Decimals: Consistent precision (usually 1-2 decimal places)
- Large numbers: Consider thousands separator

### Missing Data
- Use `-` for not applicable
- Use `NaN` or empty for missing data
- Document in caption

### Table Size
- **Main text:** < 1 page (aim for 15-20 rows max)
- **Supplementary:** Can be longer (use `longtable`)
- Split large tables into multiple smaller ones

### Captions
- Descriptive and self-contained
- Define all abbreviations
- Explain any special formatting
- Include statistical details (e.g., "Mean ± SD across 24 folds")

---

## 📚 **RESOURCES**

- [LaTeX tables guide](https://www.overleaf.com/learn/latex/Tables)
- [pandas.to_latex documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_latex.html)
- [Nature table guidelines](https://www.nature.com/nature/for-authors/formatting-guide)
- [PLOS table guidelines](https://journals.plos.org/plosone/s/tables)

---

**Next Steps:**
1. Collect real data from your `results/` directory
2. Populate CSV templates
3. Run `generate_publication_tables.py`
4. Review LaTeX output
5. Incorporate into manuscript

**Questions?** See main `V3_FINAL_IMPROVEMENTS.md` for context!

