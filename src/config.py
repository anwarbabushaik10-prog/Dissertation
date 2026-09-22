"""
Central configuration for the Explainable Plant Disease Detection project.

Keeping every path, hyperparameter and constant in one place makes the
pipeline easy to reproduce and easy to explain in the dissertation write-up.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"                     # extracted Kaggle folders live here
COLOR_DIR = RAW_DIR / "PlantVillage Dataset (Labeled)" / "Color Images"
SEGMENTED_DIR = RAW_DIR / "PlantVillage Dataset (Labeled)" / "Segmented Images"

SPLITS_DIR = DATA_ROOT / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
EXPLANATIONS_DIR = OUTPUTS_DIR / "explanations"

for d in [SPLITS_DIR, FIGURES_DIR, METRICS_DIR, CHECKPOINTS_DIR, EXPLANATIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility / scope control
# ---------------------------------------------------------------------------
SEED = 42

# NOTE ON DATASET SCOPE: the Kaggle "PlantVillage Dataset (Labeled)" package
# advertises 25 classes / 31,397 color images in its description, but the
# "Color Images" folder actually shipped inside the archive only contains
# 19 classes / 15,915 images (6 Tomato/Potato classes are missing from the
# color set, though present in "Grayscale Images"). This was verified by
# listing the archive after download (see data/raw/CLASS_LIST.txt) and is
# documented as a dataset-quality finding in RESULTS.md rather than worked
# around, per the proposal's "exact class counts will be recorded after
# download" scoping note. We train on the 19 classes that are actually
# available as color images.
#
# 15,915 images is small enough to use in full (max class = 1,799 images),
# so no per-class subsampling is required. Kept as a safety valve.
MAX_IMAGES_PER_CLASS = None

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 4

EPOCHS = 12
PATIENCE = 3             # early stopping patience (epochs without val-F1 improvement)
LR_HEAD = 5e-4           # conservative LR for the newly-initialised classifier head
LR_BACKBONE = 5e-5       # lower LR for fine-tuned pretrained layers
WEIGHT_DECAY = 5e-4      # modest regularisation for the controlled dataset
LABEL_SMOOTHING = 0.1
DROPOUT = 0.25
FREEZE_BACKBONE_EPOCHS = 2  # warm-up epochs training only the head

CNN_BACKBONE = "mobilenetv2_100"
VIT_BACKBONE = "vit_tiny_patch16_224"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------
N_EXPLAIN_SAMPLES_PER_CLASS = 2   # kept small: LIME is slow, this is a demo/eval set
BACKGROUND_FOCUS_THRESHOLD = 0.5  # >50% of heatmap mass outside the leaf mask => "misleading"
