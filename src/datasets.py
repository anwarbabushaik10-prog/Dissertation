"""
PyTorch Dataset built from the CSV manifests produced by prepare_splits.py,
plus the torchvision transform pipelines used for training/evaluation.
"""
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def build_transforms(train: bool, perturbation: str | None = None) -> transforms.Compose:
    if train:
        # Moderate, label-preserving augmentation reduces reliance on the
        # controlled PlantVillage backgrounds without obscuring symptoms.
        return transforms.Compose([
            transforms.RandomResizedCrop(
                IMAGE_SIZE, scale=(0.80, 1.0), ratio=(0.90, 1.10),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    perturbations = {
        None: [],
        "brightness_low": [transforms.ColorJitter(brightness=(0.75, 0.75))],
        "brightness_high": [transforms.ColorJitter(brightness=(1.25, 1.25))],
        "contrast_low": [transforms.ColorJitter(contrast=(0.75, 0.75))],
        "mild_blur": [transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 1.0))],
    }
    if perturbation not in perturbations:
        raise ValueError(f"Unknown evaluation perturbation: {perturbation}")
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        *perturbations[perturbation],
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PlantVillageDataset(Dataset):
    """Reads (filepath, label, class_name) rows from a manifest CSV."""

    def __init__(self, manifest_csv, class_to_idx: dict, train: bool,
                 perturbation: str | None = None):
        self.df = pd.read_csv(manifest_csv)
        self.class_to_idx = class_to_idx
        self.transform = build_transforms(train, perturbation)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self._resolve_path(row["filepath"])).convert("RGB")
        img = self.transform(img)
        label = self.class_to_idx[row["class_name"]]
        return img, label

    def raw_image(self, idx):
        """Returns the un-normalised PIL image (for explanation visualisation)."""
        row = self.df.iloc[idx]
        path = self._resolve_path(row["filepath"])
        return Image.open(path).convert("RGB"), row["class_name"], str(path)

    @staticmethod
    def _resolve_path(filepath):
        from config import PROJECT_ROOT
        path = Path(filepath)
        return path if path.is_absolute() else PROJECT_ROOT / path
