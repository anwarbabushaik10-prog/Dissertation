"""Audit split manifests for visually identical rotated/flipped images.

This is a supplementary leakage check.  The main split builder already keeps
filename-linked offline variants and exact byte duplicates together.  Here a
canonical perceptual hash adds rotation/flip invariance and verifies that no
visually identical hash occurs across train, validation, and test.

Usage:
    python src/audit_splits.py
"""
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

from config import PROJECT_ROOT, SPLITS_DIR, METRICS_DIR


def canonical_phash(filepath: str) -> str:
    path = PROJECT_ROOT / filepath
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )

    hashes = []
    for base in (pixels, np.fliplr(pixels)):
        for rotations in range(4):
            transformed = np.rot90(base, rotations)
            coefficients = dctn(transformed, norm="ortho")[:8, :8]
            threshold = np.median(coefficients.ravel()[1:])
            value = 0
            for bit in (coefficients > threshold).ravel():
                value = (value << 1) | int(bit)
            hashes.append(value)
    return f"{min(hashes):016x}"


def main():
    frames = []
    for split in ("train", "val", "test"):
        frame = pd.read_csv(SPLITS_DIR / f"{split}.csv")
        frame["split"] = split
        frames.append(frame)
    manifest = pd.concat(frames, ignore_index=True)
    manifest["canonical_phash"] = [canonical_phash(path) for path in manifest["filepath"]]

    groups = manifest.groupby("canonical_phash").agg(
        n_images=("filepath", "size"),
        n_splits=("split", "nunique"),
        n_source_groups=("source_group", "nunique"),
    )
    overlap = groups[groups["n_splits"] > 1]
    report = {
        "method": "64-bit DCT perceptual hash, canonicalised over rotations and horizontal flips",
        "n_images": int(len(manifest)),
        "perceptual_hash_groups": int(len(groups)),
        "groups_spanning_splits": int(len(overlap)),
        "images_in_spanning_groups": int(overlap["n_images"].sum()),
        "checks_passed": bool(overlap.empty),
    }
    with open(METRICS_DIR / "perceptual_hash_audit.json", "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    if not report["checks_passed"]:
        raise RuntimeError("Perceptual-hash split audit failed")


if __name__ == "__main__":
    main()
