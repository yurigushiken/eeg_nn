# Plot Directories Guide

## ✅ Simple, Clean Structure (Test Performance Only!)

### `plots_outer/` - The Only Directory You Need
**Model Selection**: AUTOMATICALLY matches your `optuna_objective` setting!  
**Data**: Outer test set predictions (out-of-fold, never seen during training)  
**Evaluation**: Ensemble of K inner models (as specified by `outer_eval_mode: ensemble`)

**Files**: 
- `fold{X}_confusion.png` - per outer fold (24 folds for LOSO with 24 subjects)
- `fold{X}_curves.png` - training curves from best inner model
- `overall_confusion.png` - grand average across all outer test predictions

**Titles dynamically show**:
- When `optuna_objective: inner_mean_min_per_class_f1`: "inner-mean min-per-class-F1=X.XX"
- When `optuna_objective: inner_mean_macro_f1`: "inner-mean macro-F1=X.XX"  
- When `optuna_objective: inner_mean_acc`: "inner-mean acc=X.XX"

---

## Why Only plots_outer/?

✅ **Test performance is what matters** - this is what you report in papers  
✅ **No confusion** - only one source of truth  
✅ **No data leakage concerns** - test data completely held out  
✅ **Clean and simple** - easy to understand and explain

**Validation is for selection only** - Optuna uses inner CV to select hyperparameters, but you don't need plots of validation performance. What matters is how the selected model performs on held-out test data.

---

## Available Optimization Objectives

Set in your YAML config (e.g., `base_min.yaml`):

```yaml
optuna_objective: inner_mean_min_per_class_f1  # Options below:
```

**Options**:
1. `inner_mean_macro_f1` - Average F1 across all classes (treats all classes equally)
2. `inner_mean_min_per_class_f1` - Minimum F1 among all classes (no class left behind)
3. `inner_mean_acc` - Overall accuracy

---

## How It Works

1. **You set** `optuna_objective` in your config
2. **Optuna searches** hyperparameters to maximize that objective across inner CV folds
3. **Training uses** ensemble of K inner models to predict on outer test set
4. **plots_outer/** shows test performance with titles matching your objective
5. **No confusion** - what you optimize is what you see!

---

## Expected Output

For 24 subjects with LOSO (`n_folds: null`):
- **25 confusion matrices**: `fold1.png` through `fold24.png` + `overall_confusion.png`
- **25 curves plots**: showing training/validation curves for each fold
- **Total: 50 plot files** in `plots_outer/`

This is publication-ready test performance that directly corresponds to your optimization objective!

