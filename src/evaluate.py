"""
Evaluates a trained checkpoint on the held-out test split and produces the
quantitative evaluation artefacts named in the proposal (slide 07):
accuracy, precision/recall, macro & weighted F1, confusion matrix,
per-class false positive/negative error analysis, and a grid of
misclassified examples.

Usage:
    python src/evaluate.py --model cnn
    python src/evaluate.py --model vit
"""
import argparse
import json

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, f1_score,
)
from torch.utils.data import DataLoader

from config import (
    SPLITS_DIR, CHECKPOINTS_DIR, METRICS_DIR, FIGURES_DIR,
    IMAGENET_MEAN, IMAGENET_STD, BATCH_SIZE, NUM_WORKERS, SEED,
)
from datasets import PlantVillageDataset
from models import build_model
from utils import get_device, seed_worker, set_seed
from viz_style import apply_style, SEQUENTIAL_CMAP


def load_model(model_name, class_to_idx, device):
    ckpt = torch.load(CHECKPOINTS_DIR / f"{model_name}_best.pt", map_location=device)
    model = build_model(model_name, len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt


@torch.no_grad()
def predict_all(model, loader, device):
    all_logits, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    return torch.cat(all_logits), torch.cat(all_labels)


def bootstrap_ci(y_true, y_pred, metric, n_resamples=1000):
    """Class-stratified bootstrap 95% CI for a held-out test metric."""
    rng = np.random.RandomState(SEED)
    class_indices = [np.where(y_true == label)[0] for label in np.unique(y_true)]
    values = []
    for _ in range(n_resamples):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in class_indices
        ])
        values.append(metric(y_true[sampled], y_pred[sampled]))
    low, high = np.percentile(values, [2.5, 97.5])
    return {"lower": float(low), "upper": float(high), "n_resamples": n_resamples}


def calibration_metrics(logits, y_true, n_bins=10):
    probs = torch.softmax(logits, dim=1).numpy()
    confidence = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == y_true
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    one_hot = np.eye(probs.shape[1])[y_true]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    nll = -np.log(np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)).mean()
    return {
        "mean_confidence": float(confidence.mean()),
        "expected_calibration_error_10_bins": float(ece),
        "multiclass_brier_score": float(brier),
        "negative_log_likelihood": float(nll),
    }


def evaluate_perturbations(model, class_to_idx, device, clean_accuracy, clean_f1):
    rows = [{
        "condition": "clean", "accuracy": clean_accuracy,
        "macro_f1": clean_f1, "accuracy_drop": 0.0, "macro_f1_drop": 0.0,
    }]
    for condition in ("brightness_low", "brightness_high", "contrast_low", "mild_blur"):
        dataset = PlantVillageDataset(
            SPLITS_DIR / "test.csv", class_to_idx, train=False,
            perturbation=condition,
        )
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
        )
        logits, labels = predict_all(model, loader, device)
        labels = labels.numpy()
        predictions = logits.argmax(1).numpy()
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, average="macro")
        rows.append({
            "condition": condition,
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "accuracy_drop": float(clean_accuracy - accuracy),
            "macro_f1_drop": float(clean_f1 - macro_f1),
        })
    return rows


def plot_confusion_matrix(cm, class_names, model_name):
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(cm, cmap=SEQUENTIAL_CMAP)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix — {model_name.upper()} (test set)", fontweight="bold")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6,
                         color="white" if cm[i, j] > thresh else "#1a1a1a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model_name}_confusion_matrix.png")
    plt.close(fig)


def plot_per_class_errors(df_errors, model_name):
    apply_style()
    df_sorted = df_errors.sort_values("false_negatives", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    y_pos = np.arange(len(df_sorted))
    ax.barh(y_pos - 0.2, df_sorted["false_negatives"], height=0.4, label="False negatives", color="#0072B2")
    ax.barh(y_pos + 0.2, df_sorted["false_positives"], height=0.4, label="False positives", color="#E69F00")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["class_name"], fontsize=7)
    ax.set_xlabel("Count (test set)")
    ax.set_title(f"Per-class error analysis — {model_name.upper()}", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model_name}_error_analysis.png")
    plt.close(fig)


def denormalise(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def plot_misclassified_grid(test_ds, y_true, y_pred, idx_to_class, model_name, n=16):
    apply_style()
    wrong_idx = np.where(y_true != y_pred)[0]
    if len(wrong_idx) == 0:
        print("No misclassified examples — skipping grid.")
        return
    chosen = np.random.RandomState(0).choice(wrong_idx, size=min(n, len(wrong_idx)), replace=False)
    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 3.4))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")

    def wrap(label, width=26):
        return "\n".join(textwrap.wrap(label, width=width))

    for ax, idx in zip(axes, chosen):
        img_tensor, _ = test_ds[idx]
        img = denormalise(img_tensor)
        ax.imshow(img)
        true_c = idx_to_class[y_true[idx]]
        pred_c = idx_to_class[y_pred[idx]]
        ax.set_title(f"true: {wrap(true_c)}\npred: {wrap(pred_c)}", fontsize=7.5, color="#D55E00")
        ax.axis("off")
    fig.suptitle(f"Misclassified test examples — {model_name.upper()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model_name}_misclassified_examples.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    class_to_idx = json.load(open(SPLITS_DIR / "class_to_idx.json"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model, ckpt = load_model(args.model, class_to_idx, device)
    print(f"Loaded {args.model} checkpoint from epoch {ckpt['epoch']} (val macro-F1={ckpt['val_f1']:.4f})")

    test_ds = PlantVillageDataset(SPLITS_DIR / "test.csv", class_to_idx, train=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
    )

    logits, y_true = predict_all(model, test_loader, device)
    y_pred = logits.argmax(1).numpy()
    y_true = y_true.numpy()

    acc = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    confidence_intervals = {
        "accuracy": bootstrap_ci(y_true, y_pred, accuracy_score),
        "macro_f1": bootstrap_ci(
            y_true, y_pred,
            lambda true, pred: f1_score(true, pred, average="macro"),
        ),
    }
    calibration = calibration_metrics(logits, y_true)

    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                         output_dict=True, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, args.model)

    # per-class FP/FN error analysis + class-level counts (ties into imbalance report)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    support = cm.sum(axis=1)
    import pandas as pd
    df_errors = pd.DataFrame({
        "class_name": class_names,
        "support": support,
        "false_positives": fp,
        "false_negatives": fn,
    })
    df_errors.to_csv(METRICS_DIR / f"{args.model}_error_analysis.csv", index=False)
    plot_per_class_errors(df_errors, args.model)

    plot_misclassified_grid(test_ds, y_true, y_pred, idx_to_class, args.model)

    robustness_rows = evaluate_perturbations(
        model, class_to_idx, device, float(acc), float(f1_macro),
    )
    import pandas as pd
    pd.DataFrame(robustness_rows).to_csv(
        METRICS_DIR / f"{args.model}_robustness.csv", index=False,
    )

    metrics = {
        "model": args.model,
        "test_accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "confidence_intervals_95": confidence_intervals,
        "calibration": calibration,
        "robustness": {
            "conditions": robustness_rows,
            "worst_accuracy": min(row["accuracy"] for row in robustness_rows),
            "largest_accuracy_drop": max(row["accuracy_drop"] for row in robustness_rows),
        },
        "n_test_samples": int(len(y_true)),
        "per_class_report": report_dict,
        "checkpoint_epoch": ckpt["epoch"],
        "checkpoint_val_f1": ckpt["val_f1"],
    }
    with open(METRICS_DIR / f"{args.model}_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n=== {args.model.upper()} test results ===")
    print(f"Accuracy:            {acc:.4f}")
    print(f"Macro  P/R/F1:       {precision_macro:.4f} / {recall_macro:.4f} / {f1_macro:.4f}")
    print(f"Weighted P/R/F1:     {precision_weighted:.4f} / {recall_weighted:.4f} / {f1_weighted:.4f}")
    print(f"Accuracy 95% CI:     {confidence_intervals['accuracy']['lower']:.4f}"
          f"–{confidence_intervals['accuracy']['upper']:.4f}")
    print(f"Worst mild-shift acc:{min(row['accuracy'] for row in robustness_rows):.4f}")
    print("Saved: confusion matrix, error analysis, misclassified grid, "
          "robustness CSV, test_metrics.json")


if __name__ == "__main__":
    main()
