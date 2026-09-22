"""
Scans the extracted Color Images folder, builds a stratified 70/15/15
train/val/test split per class, writes manifest CSVs, and produces the
class-imbalance figure/table referenced in the proposal's evaluation
framework (slide 06/07: "class imbalance report", "stratified split").

Usage:
    python src/prepare_splits.py
"""
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import (
    PROJECT_ROOT, COLOR_DIR, SPLITS_DIR, FIGURES_DIR, METRICS_DIR,
    SEED, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, MAX_IMAGES_PER_CLASS,
)
from viz_style import apply_style, CATEGORICAL


def scan_color_images():
    assert COLOR_DIR.exists(), f"Color Images folder not found at {COLOR_DIR}"
    rows = []
    for class_dir in sorted(COLOR_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append({
                    "filepath": str(f.relative_to(PROJECT_ROOT)),
                    "class_name": class_dir.name,
                })
    return pd.DataFrame(rows)


def source_identity(filepath: str) -> str:
    """Recover the source-image identifier embedded in PlantVillage names.

    Several classes contain offline augmented files such as ``copy 2`` with
    a different UUID prefix.  Grouping on the suffix after ``___`` keeps all
    variants of the same leaf in one partition.
    """
    stem = Path(filepath).stem.lower()
    stem = stem.split("___", 1)[-1]
    stem = re.sub(r"\s*copy(?:\s*\d+)?\s*$", "", stem)
    return re.sub(r"[^a-z0-9]+", " ", stem).strip()


def sha256_file(filepath: str) -> str:
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[right] = left


def attach_source_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe groups using both filename identity and hashes."""
    df = df.reset_index(drop=True).copy()
    df["source_identity"] = df["filepath"].map(source_identity)
    df["sha256"] = [sha256_file(path) for path in df["filepath"]]

    # An exact image appearing under two labels is a dataset error, not a
    # grouping problem that should silently be hidden.
    conflicting = df.groupby("sha256")["class_name"].nunique()
    if (conflicting > 1).any():
        raise ValueError("Identical image bytes occur under different class labels")

    dsu = DisjointSet(len(df))
    first_by_identity, first_by_hash = {}, {}
    for idx, row in df.iterrows():
        identity_key = (row["class_name"], row["source_identity"])
        if identity_key in first_by_identity:
            dsu.union(idx, first_by_identity[identity_key])
        else:
            first_by_identity[identity_key] = idx
        if row["sha256"] in first_by_hash:
            dsu.union(idx, first_by_hash[row["sha256"]])
        else:
            first_by_hash[row["sha256"]] = idx

    members = defaultdict(list)
    for idx in range(len(df)):
        members[dsu.find(idx)].append(df.loc[idx, "filepath"])
    group_ids = {}
    for root, paths in members.items():
        value = "\n".join(sorted(paths)).encode("utf-8")
        group_ids[root] = hashlib.sha1(value).hexdigest()[:16]
    df["source_group"] = [group_ids[dsu.find(idx)] for idx in range(len(df))]
    return df


def stratified_group_split(df: pd.DataFrame, seed: int):
    """70/15/15 class-stratified split with source groups kept intact."""
    rng = random.Random(seed)
    train_rows, val_rows, test_rows = [], [], []

    for class_name, group in df.groupby("class_name"):
        source_groups = [part for _, part in group.groupby("source_group")]
        rng.shuffle(source_groups)
        if MAX_IMAGES_PER_CLASS is not None:
            kept, count = [], 0
            for source_group in source_groups:
                if count >= MAX_IMAGES_PER_CLASS:
                    break
                kept.append(source_group)
                count += len(source_group)
            source_groups = kept

        n = sum(len(part) for part in source_groups)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        cumulative = 0
        for source_group in source_groups:
            if cumulative < n_train:
                train_rows.append(source_group)
            elif cumulative < n_train + n_val:
                val_rows.append(source_group)
            else:
                test_rows.append(source_group)
            cumulative += len(source_group)

    train_df = pd.concat(train_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    val_df = pd.concat(val_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, val_df, test_df


def write_integrity_report(full_df, split_frames):
    membership = pd.concat([
        frame.assign(split=name) for name, frame in split_frames.items()
    ], ignore_index=True)
    group_overlap = membership.groupby("source_group")["split"].nunique()
    hash_overlap = membership.groupby("sha256")["split"].nunique()
    report = {
        "strategy": "class-stratified source-group split (70/15/15)",
        "source_group_rule": "normalised filename suffix after ___ plus exact SHA-256 linkage",
        "n_images": int(len(membership)),
        "n_source_groups": int(full_df["source_group"].nunique()),
        "offline_variant_groups": int((full_df.groupby("source_group").size() > 1).sum()),
        "source_groups_spanning_splits": int((group_overlap > 1).sum()),
        "exact_hashes_spanning_splits": int((hash_overlap > 1).sum()),
        "split_sizes": {name: int(len(frame)) for name, frame in split_frames.items()},
        "checks_passed": bool((group_overlap <= 1).all() and (hash_overlap <= 1).all()),
    }
    with open(METRICS_DIR / "split_integrity_report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    if not report["checks_passed"]:
        raise RuntimeError("Split integrity check failed")
    return report


def plot_class_distribution(df: pd.DataFrame, split_counts: dict):
    apply_style()
    counts = df["class_name"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    bars = ax.barh(counts.index, counts.values, color=CATEGORICAL[0], height=0.65)
    ax.set_xlabel("Number of images")
    ax.set_title("PlantVillage (Color Images) — class distribution\n"
                  f"{len(df)} images across {df['class_name'].nunique()} classes "
                  "(imbalanced, ratio "
                  f"{counts.max()}:{counts.min()} = {counts.max()/counts.min():.1f}x)")
    for bar, val in zip(bars, counts.values):
        ax.text(val + max(counts.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8, color="#595959")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_distribution.png")
    plt.close(fig)


def main():
    print(f"Scanning {COLOR_DIR} ...")
    df = attach_source_groups(scan_color_images())
    print(f"Found {len(df)} images across {df['class_name'].nunique()} classes.")

    classes = sorted(df["class_name"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df, val_df, test_df = stratified_group_split(df, SEED)

    integrity = write_integrity_report(df, {
        "train": train_df, "val": val_df, "test": test_df,
    })
    print("Split integrity:", integrity)

    manifest_columns = ["filepath", "class_name", "source_group"]
    train_df[manifest_columns].to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df[manifest_columns].to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df[manifest_columns].to_csv(SPLITS_DIR / "test.csv", index=False)
    with open(SPLITS_DIR / "class_to_idx.json", "w") as f:
        json.dump(class_to_idx, f, indent=2)

    split_counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    print("Split sizes:", split_counts)

    # Class imbalance report (counts + train/val/test breakdown per class)
    report = (
        df.groupby("class_name").size().rename("total")
        .to_frame()
        .join(train_df.groupby("class_name").size().rename("train"))
        .join(val_df.groupby("class_name").size().rename("val"))
        .join(test_df.groupby("class_name").size().rename("test"))
        .fillna(0).astype(int)
        .sort_values("total", ascending=False)
    )
    report.to_csv(METRICS_DIR / "class_imbalance_report.csv")
    print("\nClass imbalance report:\n", report)

    plot_class_distribution(df, split_counts)

    summary = {
        "n_images_total": int(len(df)),
        "n_classes": int(df["class_name"].nunique()),
        "split_sizes": split_counts,
        "max_class_count": int(report["total"].max()),
        "min_class_count": int(report["total"].min()),
        "imbalance_ratio": round(report["total"].max() / report["total"].min(), 2),
        "classes": classes,
    }
    with open(METRICS_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved manifests to", SPLITS_DIR)
    print("Saved figure to", FIGURES_DIR / "class_distribution.png")


if __name__ == "__main__":
    main()
