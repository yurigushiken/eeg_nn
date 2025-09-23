from __future__ import annotations
import json
from pathlib import Path
from utils.reporting import create_consolidated_reports
import yaml


def write_summary(run_dir: Path, summary: dict, task: str, engine: str):
    """Write summary JSON and a detailed TXT report."""
    
    # Write raw JSON summary
    out_fp_json = run_dir / f"summary_{task}_{engine}.json"
    out_fp_json.write_text(json.dumps(summary, indent=2))

    # Also write resolved config (hyper) as YAML for easy reuse
    try:
        hyper = summary.get("hyper") or {}
        if isinstance(hyper, dict):
            out_fp_yaml = run_dir / "resolved_config.yaml"
            out_fp_yaml.write_text(yaml.safe_dump(hyper, sort_keys=False))
    except Exception:
        # keep summary generation resilient
        pass
    
    # --- Build and Write Detailed TXT Report ---
    report_lines = []
    
    # Header
    report_lines.append(f"--- Run Summary: {task} ({engine}) ---")
    report_lines.append(f"Run Directory: {run_dir.name}")
    report_lines.append("")
    
    # Overall Metrics
    report_lines.append("--- Overall Metrics ---")
    report_lines.append(f"Mean Accuracy: {summary.get('mean_acc', 0.0):.2f}% (std: {summary.get('std_acc', 0.0):.2f})")
    report_lines.append(f"Macro F1-Score: {summary.get('macro_f1', 0.0):.2f}")
    report_lines.append(f"Weighted F1-Score: {summary.get('weighted_f1', 0.0):.2f}")
    report_lines.append(f"Inner Val Mean Acc: {summary.get('inner_mean_acc', 0.0):.2f}%")
    report_lines.append(f"Inner Val Mean Macro F1: {summary.get('inner_mean_macro_f1', 0.0):.2f}")
    report_lines.append("")

    # Fold Breakdown
    report_lines.append("--- Fold Breakdown ---")
    if summary.get("fold_splits"):
        for i, split in enumerate(summary["fold_splits"]):
            acc = summary["fold_accuracies"][i] if i < len(summary.get("fold_accuracies", [])) else "N/A"
            subjects = ", ".join(map(str, split['test_subjects']))
            report_lines.append(f"  Fold {split['fold']:02d} | Test Subjects: [{subjects}] | Accuracy: {acc:.2f}%")
    report_lines.append("")

    # Classification Report
    if summary.get("classification_report"):
        report_lines.append("--- Classification Report ---")
        report_lines.append(summary["classification_report"])
        report_lines.append("")

    # Hyperparameters
    report_lines.append("--- Hyperparameters ---")
    if summary.get("hyper"):
        for key, value in summary["hyper"].items():
            report_lines.append(f"  {key}: {value}")
    report_lines.append("")
    
    # Write report
    out_fp_txt = run_dir / f"report_{task}_{engine}.txt"
    out_fp_txt.write_text("\n".join(report_lines))
    print(f"  [summary] done · mean_acc={summary.get('mean_acc', 0.0):.2f}%")

    # Generate consolidated HTML and PDF reports
    try:
        print("\n--- Generating Consolidated Reports ---")
        create_consolidated_reports(run_dir, summary, task, engine)
    except Exception as e:
        print(f" !! ERROR generating consolidated reports: {e}")


