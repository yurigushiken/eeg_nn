from __future__ import annotations
import json
import hashlib
from pathlib import Path
try:
    from utils.reporting import create_consolidated_reports
except Exception:
    def create_consolidated_reports(run_dir: Path, summary: dict, task: str, engine: str):
        return None
import yaml
import sys

def _compute_chance_level(num_classes: int | None) -> float:
    try:
        if num_classes and num_classes > 0:
            return 100.0 / float(num_classes)
    except Exception:
        pass
    return 0.0

def _compute_config_hash(hyper: dict) -> str:
    """Compute a deterministic hash of hyperparameters for tracking/caching."""
    try:
        # Sort keys for determinism
        serialized = json.dumps(hyper, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    except Exception:
        return "unknown"

def _collect_lib_versions() -> dict:
    versions = {}
    try:
        import torch
        versions["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import numpy as np
        versions["numpy"] = np.__version__
    except Exception:
        pass
    try:
        import sklearn
        versions["sklearn"] = sklearn.__version__
    except Exception:
        pass
    try:
        import mne
        versions["mne"] = mne.__version__
    except Exception:
        pass
    try:
        import braindecode
        versions["braindecode"] = braindecode.__version__
    except Exception:
        pass
    try:
        import captum
        versions["captum"] = captum.__version__
    except Exception:
        pass
    try:
        import optuna
        versions["optuna"] = optuna.__version__
    except Exception:
        pass
    versions["python"] = sys.version.split(" ")[0]
    return versions

def _collect_hardware_info() -> dict:
    info = {}
    try:
        import torch
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            try:
                info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            except Exception:
                info["gpu_names"] = []
    except Exception:
        pass
    try:
        import platform
        info["cpu"] = platform.processor() or platform.machine()
        info["system"] = platform.platform()
    except Exception:
        pass
    return info

"""
Summary writer: persists metrics/artifacts for a single run.

Writes into the run directory:
- summary_TASK_ENGINE.json (raw dict)
- resolved_config.yaml (frozen hyperparameters for reproducibility)
- report_TASK_ENGINE.txt (human-readable summary)
Then calls create_consolidated_reports to generate HTML/PDF summaries.

Resiliency: failures in any artifact generation step should not crash runs.
"""


def write_summary(run_dir: Path, summary: dict, task: str, engine: str):
    """Write summary JSON, resolved YAML, TXT report, and HTML/PDF.

    Keep this resilient: a failure in one artifact should not block others.
    """
    # Collect library versions for provenance (added to JSON/TXT banners)
    lib_versions = _collect_lib_versions()
    summary.setdefault("lib_versions", lib_versions)
    # Collect hardware/runtime info
    hw_info = _collect_hardware_info()
    summary.setdefault("hardware", hw_info)
    
    # Add config hash for tracking/caching (T023)
    try:
        hyper = summary.get("hyper", {})
        if isinstance(hyper, dict) and hyper:
            summary["config_hash"] = _compute_config_hash(hyper)
    except Exception:
        pass
    
    # Include exclusion summary if dataset excluded subjects (T023)
    try:
        excluded_subjects = summary.get("excluded_subjects") or {}
        if excluded_subjects:
            summary["exclusion_summary"] = {
                "count": len(excluded_subjects),
                "subject_ids": sorted(excluded_subjects.keys()),
            }
    except Exception:
        pass
    
    # Reference runtime telemetry log (T023)
    try:
        runtime_log = run_dir / "logs" / "runtime.jsonl"
        if runtime_log.exists():
            summary["runtime_log_path"] = str(runtime_log.relative_to(run_dir.parent.parent))
    except Exception:
        pass

    # Write raw JSON summary
    out_fp_json = run_dir / f"summary_{task}_{engine}.json"
    out_fp_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also write resolved config (hyper) as YAML for easy reuse (omit env/path keys)
    try:
        hyper = summary.get("hyper") or {}
        if isinstance(hyper, dict):
            out_fp_yaml = run_dir / "resolved_config.yaml"
            out_fp_yaml.write_text(yaml.safe_dump(hyper, sort_keys=False), encoding="utf-8")
    except Exception:
        # keep summary generation resilient
        pass

    # --- Build and Write Detailed TXT Report ---
    report_lines = []

    # Header
    report_lines.append(f"--- Run Summary: {task} ({engine}) ---")
    report_lines.append(f"Run Directory: {run_dir.name}")
    
    # Command line (if available)
    if summary.get("command"):
        report_lines.append(f"Command: {summary['command']}")
    
    report_lines.append("")

    # Environment banner
    model_class = summary.get("model_class") or "(unknown)"
    model_sig = summary.get("model_input_signature") or "(unknown)"
    det = summary.get("determinism") or {}
    report_lines.append("--- Environment ---")
    report_lines.append(f"Model: {model_class} (input={model_sig})")
    if lib_versions:
        libs_str = ", ".join([f"{k} {v}" for k, v in lib_versions.items()])
        report_lines.append(f"Libs: {libs_str}")
    if hw_info:
        try:
            gpus = ", ".join(hw_info.get("gpu_names", []))
            report_lines.append(f"Hardware: CUDA={hw_info.get('cuda_available')} cuda_version={hw_info.get('cuda_version')} GPUs=[{gpus}] CPU={hw_info.get('cpu')} {hw_info.get('system')}")
        except Exception:
            pass
    if det:
        report_lines.append("Determinism:")
        for k, v in det.items():
            report_lines.append(f"  {k}: {v}")
    report_lines.append("")

    # Overall Metrics (restructured for scientific clarity)
    report_lines.append("--- Overall Metrics ---")
    report_lines.append("")
    
    # 1. REFERENCE BASELINE
    report_lines.append("REFERENCE BASELINE:")
    try:
        ch = _compute_chance_level(summary.get("num_classes"))
        if ch > 0.0:
            n_classes = summary.get("num_classes", 0)
            report_lines.append(f"  Chance Level: {ch:.2f}% ({n_classes}-class balanced)")
        else:
            report_lines.append(f"  Chance Level: Unknown")
    except Exception:
        report_lines.append(f"  Chance Level: Unknown")
    report_lines.append("")
    
    # 2. INNER VALIDATION (Model Learning)
    report_lines.append("INNER VALIDATION (Model Learning):")
    report_lines.append(f"  Mean Accuracy: {summary.get('inner_mean_acc', 0.0):.2f}%")
    report_lines.append(f"  Mean Macro F1: {summary.get('inner_mean_macro_f1', 0.0):.2f}")
    if summary.get('inner_mean_min_per_class_f1') is not None:
        report_lines.append(f"  Mean Min-Per-Class F1: {summary.get('inner_mean_min_per_class_f1', 0.0):.2f}")
    if summary.get('inner_mean_plur_corr') is not None:
        report_lines.append(f"  Mean Plurality Correctness: {summary.get('inner_mean_plur_corr', 0.0):.2f}%")
    
    # Show composite objective interpretation if applicable
    if summary.get('composite_min_f1_plur_corr') is not None:
        composite_val = summary.get('composite_min_f1_plur_corr', 0.0)
        inner_min_f1 = summary.get('inner_mean_min_per_class_f1', 0.0)
        inner_plur = summary.get('inner_mean_plur_corr', 0.0)
        # Infer threshold from config (assuming 35-38 for 3-class, 20 for 6-class)
        # If inner_min_f1 >= 35, composite = plurality; else composite = min_f1 * 0.1
        if composite_val > 10.0:  # Likely above threshold
            report_lines.append(f"  → Composite Objective: {composite_val:.2f} (threshold met, optimizing distinctness)")
        else:  # Below threshold
            report_lines.append(f"  → Composite Objective: {composite_val:.2f} (below threshold, gradient penalty)")
    report_lines.append("")
    
    # 3. OUTER TEST (Cross-Subject Generalization)
    report_lines.append("OUTER TEST (Cross-Subject Generalization):")
    report_lines.append(f"  Mean Accuracy: {summary.get('mean_acc', 0.0):.2f}% ± {summary.get('std_acc', 0.0):.2f}")
    report_lines.append(f"  Mean Macro F1: {summary.get('macro_f1', 0.0):.2f}")
    if summary.get('mean_min_per_class_f1') is not None:
        report_lines.append(f"  Mean Min-Per-Class F1: {summary.get('mean_min_per_class_f1', 0.0):.2f} ± {summary.get('std_min_per_class_f1', 0.0):.2f}")
    if summary.get('mean_plur_corr') is not None:
        report_lines.append(f"  Mean Plurality Correctness: {summary.get('mean_plur_corr', 0.0):.2f}% ± {summary.get('std_plur_corr', 0.0):.2f}")
    
    # Compute outer composite for interpretation
    if summary.get('mean_min_per_class_f1') is not None and summary.get('mean_plur_corr') is not None:
        outer_min_f1 = summary.get('mean_min_per_class_f1', 0.0)
        outer_plur = summary.get('mean_plur_corr', 0.0)
        # Infer threshold (heuristic: if inner composite > 10, threshold was met)
        inner_composite = summary.get('composite_min_f1_plur_corr', 0.0)
        if inner_composite > 10.0:  # Threshold likely 35-38
            estimated_threshold = 35.0
        else:
            estimated_threshold = 20.0
        
        if outer_min_f1 >= estimated_threshold:
            outer_composite = outer_plur
            report_lines.append(f"  → Composite Objective: {outer_composite:.2f} (threshold met, distinctness={outer_plur:.2f}%)")
        else:
            outer_composite = outer_min_f1 * 0.1
            report_lines.append(f"  → Composite Objective: {outer_composite:.2f} (below threshold, gradient penalty)")
    report_lines.append("")
    
    # 4. GENERALIZATION GAP (Inner → Outer)
    report_lines.append("GENERALIZATION GAP (Inner → Outer):")
    inner_acc = summary.get('inner_mean_acc', 0.0)
    outer_acc = summary.get('mean_acc', 0.0)
    acc_gap = outer_acc - inner_acc
    report_lines.append(f"  Accuracy: {acc_gap:+.2f} pp {'(minimal degradation)' if abs(acc_gap) < 3.0 else '(moderate degradation)' if abs(acc_gap) < 8.0 else '(substantial degradation)'}")
    
    if summary.get('inner_mean_min_per_class_f1') is not None and summary.get('mean_min_per_class_f1') is not None:
        inner_min_f1 = summary.get('inner_mean_min_per_class_f1', 0.0)
        outer_min_f1 = summary.get('mean_min_per_class_f1', 0.0)
        min_f1_gap = outer_min_f1 - inner_min_f1
        report_lines.append(f"  Min-Per-Class F1: {min_f1_gap:+.2f} pp {'(minimal degradation)' if abs(min_f1_gap) < 3.0 else '(moderate degradation)' if abs(min_f1_gap) < 8.0 else '(substantial degradation)'}")
    
    if summary.get('inner_mean_plur_corr') is not None and summary.get('mean_plur_corr') is not None:
        inner_plur = summary.get('inner_mean_plur_corr', 0.0)
        outer_plur = summary.get('mean_plur_corr', 0.0)
        plur_gap = outer_plur - inner_plur
        report_lines.append(f"  Plurality Correctness: {plur_gap:+.2f} pp {'(minimal degradation)' if abs(plur_gap) < 10.0 else '(moderate degradation)' if abs(plur_gap) < 25.0 else '(substantial degradation)'}")
    report_lines.append("")
    
    # 5. KEY FINDING (Interpretation)
    report_lines.append("KEY FINDING:")
    if summary.get('inner_mean_plur_corr') is not None and summary.get('mean_plur_corr') is not None:
        inner_plur = summary.get('inner_mean_plur_corr', 0.0)
        outer_plur = summary.get('mean_plur_corr', 0.0)
        if inner_plur > 70.0 and outer_plur < inner_plur - 15.0:
            report_lines.append(f"  Models learn distinct patterns within subjects ({inner_plur:.2f}% plurality)")
            report_lines.append(f"  but show reduced distinctness across subjects ({outer_plur:.2f}% plurality),")
            report_lines.append(f"  suggesting partially subject-specific neural signatures.")
        elif inner_plur > 70.0 and outer_plur > 70.0:
            report_lines.append(f"  Models learn distinct patterns ({inner_plur:.2f}% plurality) that")
            report_lines.append(f"  generalize well across subjects ({outer_plur:.2f}% plurality),")
            report_lines.append(f"  suggesting robust subject-invariant neural signatures.")
        else:
            report_lines.append(f"  Models show moderate pattern distinctness (inner: {inner_plur:.2f}%,")
            report_lines.append(f"  outer: {outer_plur:.2f}%), suggesting neural signatures are")
            report_lines.append(f"  detectable but not strongly distinct across numerosities.")
    else:
        report_lines.append(f"  Mean accuracy: {outer_acc:.2f}% (vs. {ch:.2f}% chance)")
    report_lines.append("")
    
    # Additional legacy metrics for completeness
    report_lines.append("ADDITIONAL METRICS:")
    report_lines.append(f"  Weighted F1-Score: {summary.get('weighted_f1', 0.0):.2f}")
    report_lines.append(f"  Cohen's Kappa: {summary.get('mean_kappa', 0.0):.3f} ± {summary.get('std_kappa', 0.0):.3f}")
    report_lines.append("")
    
    # Decision Layer Analysis (if enabled)
    if summary.get("decision_layer_stats"):
        try:
            from code.posthoc.decision_layer import format_decision_layer_txt_section, build_decision_layer_json_fields
            dl_stats = summary["decision_layer_stats"]
            
            # Add TXT section
            dl_txt_lines = format_decision_layer_txt_section(dl_stats)
            report_lines.extend(dl_txt_lines)
            
            # Merge JSON fields into summary
            dl_json_fields = build_decision_layer_json_fields(dl_stats)
            summary.update(dl_json_fields)
        except Exception as e:
            report_lines.append(f"  [Decision Layer section unavailable: {e}]")
            report_lines.append("")

    # Fold Breakdown
    report_lines.append("--- Fold Breakdown ---")
    if summary.get("fold_splits"):
        for i, split in enumerate(summary["fold_splits"]):
            acc = summary["fold_accuracies"][i] if i < len(summary.get("fold_accuracies", [])) else None
            subjects = ", ".join(map(str, split['test_subjects']))
            if isinstance(acc, (int, float)):
                acc_str = f"{acc:.2f}%"
            else:
                acc_str = "N/A"
            report_lines.append(f"  Fold {split['fold']:02d} | Test Subjects: [{subjects}] | Accuracy: {acc_str}")
    report_lines.append("")

    # Hyperparameters (flat listing for quick inspection)
    report_lines.append("--- Hyperparameters ---")
    if summary.get("config_hash"):
        report_lines.append(f"Config Hash: {summary['config_hash']}")
    if summary.get("hyper"):
        for key, value in summary["hyper"].items():
            report_lines.append(f"  {key}: {value}")
    report_lines.append("")
    
    # Exclusion Summary (T023)
    if summary.get("exclusion_summary"):
        report_lines.append("--- Dataset Exclusions ---")
        excl = summary["exclusion_summary"]
        report_lines.append(f"Excluded Subjects: {excl['count']} subject(s)")
        report_lines.append(f"IDs: {excl['subject_ids']}")
        report_lines.append("")
    
    # Runtime Telemetry Reference (T023)
    if summary.get("runtime_log_path"):
        report_lines.append("--- Runtime Telemetry ---")
        report_lines.append(f"JSONL Log: {summary['runtime_log_path']}")
        report_lines.append("")

    # Write report
    out_fp_txt = run_dir / f"report_{task}_{engine}.txt"
    out_fp_txt.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  [summary] done · mean_acc={summary.get('mean_acc', 0.0):.2f}%")

    # Generate consolidated HTML and PDF reports (PDF optional via Playwright)
    try:
        print("\n--- Generating Consolidated Reports ---")
        create_consolidated_reports(run_dir, summary, task, engine)
    except Exception as e:
        print(f" !! ERROR generating consolidated reports: {e}")


