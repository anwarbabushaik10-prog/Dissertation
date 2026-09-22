"""Export the final CNN/ViT results as one report-ready CSV table."""
import json

import pandas as pd

from config import METRICS_DIR


def load_json(name):
    with open(METRICS_DIR / name) as handle:
        return json.load(handle)


def main():
    rows = []
    for model in ("cnn", "vit"):
        metrics = load_json(f"{model}_test_metrics.json")
        explanations = load_json(f"{model}_explanation_summary.json")
        rows.append({
            "model": model,
            "test_accuracy": metrics["test_accuracy"],
            "accuracy_ci_95_lower": metrics["confidence_intervals_95"]["accuracy"]["lower"],
            "accuracy_ci_95_upper": metrics["confidence_intervals_95"]["accuracy"]["upper"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "worst_mild_shift_accuracy": metrics["robustness"]["worst_accuracy"],
            "calibration_ece_10_bins": metrics["calibration"]["expected_calibration_error_10_bins"],
            "gradcam_foreground_fraction": explanations["gradcam_mean_foreground_fraction"],
            "gradcam_pct_misleading": explanations["gradcam_pct_misleading"],
            "lime_foreground_fraction": explanations["lime_mean_foreground_fraction"],
            "lime_pct_misleading": explanations["lime_pct_misleading"],
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
