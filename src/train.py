"""
Trains either the transfer-learning CNN or the lightweight ViT on the
PlantVillage Color Images split. Both models share the same training
recipe so the comparison in the dissertation is fair: same splits, same
augmentation, same optimiser schedule (frozen-backbone warm-up then
low-LR fine-tuning), same early-stopping criterion (macro F1 on val).

Usage:
    python src/train.py --model cnn
    python src/train.py --model vit
"""
import argparse
import json
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import (
    SPLITS_DIR, CHECKPOINTS_DIR, METRICS_DIR, FIGURES_DIR,
    BATCH_SIZE, NUM_WORKERS, EPOCHS, PATIENCE,
    LR_HEAD, LR_BACKBONE, WEIGHT_DECAY, FREEZE_BACKBONE_EPOCHS, SEED,
    LABEL_SMOOTHING,
)
from datasets import PlantVillageDataset
from models import build_model, set_backbone_trainable, param_groups
from utils import set_seed, seed_worker, get_device
from viz_style import apply_style, CATEGORICAL


def make_sampler(df, class_to_idx, generator):
    """Inverse-frequency WeightedRandomSampler: mitigates the class
    imbalance documented in class_imbalance_report.csv by up-weighting
    rare classes during training, per the proposal's risk mitigation
    ("stratified split, augmentation and macro/weighted metrics")."""
    labels = df["class_name"].map(class_to_idx).values
    class_counts = df["class_name"].value_counts()
    class_weight = {class_to_idx[c]: 1.0 / n for c, n in class_counts.items()}
    sample_weights = [class_weight[label] for label in labels]
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True,
        generator=generator,
    )


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.argmax(1).detach().cpu())
            all_labels.append(y.detach().cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def plot_curves(history, model_name):
    apply_style()
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, history["train_loss"], label="train", color=CATEGORICAL[0], linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="val", color=CATEGORICAL[1], linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend(frameon=False)

    axes[1].plot(epochs, history["train_f1"], label="train macro-F1", color=CATEGORICAL[0], linewidth=2)
    axes[1].plot(epochs, history["val_f1"], label="val macro-F1", color=CATEGORICAL[1], linewidth=2)
    axes[1].set_title("Macro F1-score")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Training curves — {model_name.upper()}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{model_name}_training_curves.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    print(f"Training '{args.model}' on device={device}")

    class_to_idx = json.load(open(SPLITS_DIR / "class_to_idx.json"))
    num_classes = len(class_to_idx)

    train_ds = PlantVillageDataset(SPLITS_DIR / "train.csv", class_to_idx, train=True)
    val_ds = PlantVillageDataset(SPLITS_DIR / "val.csv", class_to_idx, train=False)

    generator = torch.Generator().manual_seed(SEED)
    sampler = make_sampler(train_ds.df, class_to_idx, generator)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                               num_workers=NUM_WORKERS, drop_last=True,
                               worker_init_fn=seed_worker, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                             generator=generator)

    model = build_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "val_acc": []}
    best_f1, best_epoch, epochs_without_improve = -1.0, -1, 0
    ckpt_path = CHECKPOINTS_DIR / f"{args.model}_best.pt"

    set_backbone_trainable(model, args.model, trainable=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY,
    )

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        # Warm-up: train classifier head only for the first FREEZE_BACKBONE_EPOCHS,
        # then unfreeze the backbone at a lower LR for fine-tuning.
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            set_backbone_trainable(model, args.model, trainable=True)
            optimizer = torch.optim.AdamW(
                param_groups(model, LR_HEAD, LR_BACKBONE), weight_decay=WEIGHT_DECAY,
            )

        t0 = time.time()
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        stage = "warmup(head-only)" if epoch <= FREEZE_BACKBONE_EPOCHS else "finetune"
        print(f"[{args.model}] epoch {epoch:02d}/{args.epochs} ({stage}, {dt:.1f}s) "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, best_epoch, epochs_without_improve = val_f1, epoch, 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_f1": val_f1, "val_acc": val_acc,
                        "class_to_idx": class_to_idx, "model_name": args.model},
                       ckpt_path)
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val-F1 improvement for {PATIENCE} epochs).")
                break

    total_time = time.time() - t_start
    print(f"Done in {total_time/60:.1f} min. Best val macro-F1={best_f1:.4f} at epoch {best_epoch}. "
          f"Checkpoint: {ckpt_path}")

    history["best_epoch"] = best_epoch
    history["best_val_f1"] = best_f1
    history["train_time_seconds"] = total_time
    with open(METRICS_DIR / f"{args.model}_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, args.model)


if __name__ == "__main__":
    main()
