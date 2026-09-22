"""
Explainability pipeline: Grad-CAM and LIME for both the CNN and the ViT,
plus a quantitative "explanation quality" score built from the dataset's
own Segmented Images (leaf/foreground vs. black background masks).

This operationalises the proposal's evaluation framework (slide 07):
  - "Disease-region focus" / "Background false focus"  -> mask-overlap score
  - "Grad-CAM vs LIME"                                  -> both computed per image
  - "Misleading heatmap log"                            -> flagged when
                                                            background_fraction > threshold

Scope note: PlantVillage has no lesion-level (disease-region-only) masks,
only leaf-vs-background segmentation. So "disease-region focus" is
approximated as "leaf-foreground focus" (background = soil/board/plain
backdrop behind the leaf). This proxy, and its limits, are documented in
RESULTS.md rather than glossed over.

Usage:
    python src/explain.py --model cnn
    python src/explain.py --model vit
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from lime import lime_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from skimage.segmentation import mark_boundaries

from config import (
    PROJECT_ROOT, SPLITS_DIR, CHECKPOINTS_DIR, METRICS_DIR, EXPLANATIONS_DIR,
    SEGMENTED_DIR, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    N_EXPLAIN_SAMPLES_PER_CLASS, BACKGROUND_FOCUS_THRESHOLD, SEED,
)
from models import build_model

MEAN = np.array(IMAGENET_MEAN)
STD = np.array(IMAGENET_STD)


# ---------------------------------------------------------------------------
# Preprocessing helpers (kept independent of datasets.py so this script can
# feed raw numpy images to both Grad-CAM and LIME identically)
# ---------------------------------------------------------------------------
def load_and_resize(filepath):
    path = Path(filepath)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(img)  # HWC uint8


def to_model_input(img_uint8_batch, device):
    """HWC uint8 numpy (batch) -> normalised NCHW tensor."""
    x = img_uint8_batch.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float().to(device)
    return x


def find_segmented_mask(color_filepath: str, class_name: str):
    """Returns a binary foreground mask (IMAGE_SIZE x IMAGE_SIZE, {0,1}) built
    from the matching Segmented Images file, or None if unavailable (class
    not covered by the segmented set, or file naming edge-case)."""
    seg_dir = SEGMENTED_DIR / class_name
    if not seg_dir.exists():
        return None
    stem = Path(color_filepath).stem
    candidates = list(seg_dir.glob(f"{stem}_final_masked.*"))
    if not candidates:
        candidates = list(seg_dir.glob(f"{stem}*"))
    if not candidates:
        return None
    seg_img = Image.open(candidates[0]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    seg_arr = np.array(seg_img)
    # Segmented images have a pure black background; anything non-black is leaf/foreground.
    mask = (seg_arr.sum(axis=-1) > 15).astype(np.float32)
    return mask


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def vit_reshape_transform(tensor, height=IMAGE_SIZE // 16, width=IMAGE_SIZE // 16):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def build_gradcam(model, model_name):
    if model_name == "cnn":
        target_layers = [model.conv_head]
        return GradCAM(model=model, target_layers=target_layers)
    else:
        target_layers = [model.blocks[-1].norm1]
        return GradCAM(model=model, target_layers=target_layers, reshape_transform=vit_reshape_transform)


def gradcam_heatmap(cam, img_uint8, pred_class, device):
    x = to_model_input(img_uint8[None], device)
    grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_class)])[0]
    return grayscale_cam  # HxW in [0,1], already non-negative (post-ReLU)


# ---------------------------------------------------------------------------
# LIME
# ---------------------------------------------------------------------------
def make_lime_predict_fn(model, device):
    @torch.no_grad()
    def predict_fn(images_uint8_batch):
        x = to_model_input(images_uint8_batch, device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        return probs.cpu().numpy()
    return predict_fn


def lime_heatmap(explainer, predict_fn, img_uint8, pred_class, num_samples=250):
    explanation = explainer.explain_instance(
        img_uint8, predict_fn, top_labels=1, hide_color=0,
        num_samples=num_samples, segmentation_fn=None, random_seed=SEED,
    )
    label = explanation.top_labels[0]
    segments = explanation.segments
    weights = dict(explanation.local_exp[label])
    heatmap = np.zeros(segments.shape, dtype=np.float32)
    for seg_val, w in weights.items():
        heatmap[segments == seg_val] = w
    heatmap_pos = np.clip(heatmap, 0, None)  # only positive evidence counts as "focus"
    if heatmap_pos.max() > 0:
        heatmap_pos = heatmap_pos / heatmap_pos.max()
    lime_img, lime_mask = explanation.get_image_and_mask(
        label, positive_only=True, num_features=8, hide_rest=False)
    return heatmap_pos, (lime_img, lime_mask), label


# ---------------------------------------------------------------------------
# Quantitative overlap score
# ---------------------------------------------------------------------------
def overlap_score(heatmap, mask):
    total = heatmap.sum()
    if total <= 1e-8:
        return None
    fg_fraction = float((heatmap * mask).sum() / total)
    return fg_fraction


def overlay_image(img_uint8, heatmap):
    """Simple alpha-blend overlay for visualisation (no extra dependency)."""
    heat = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    cmap = plt.get_cmap("jet")
    heat_rgb = (cmap(heat)[..., :3] * 255).astype(np.uint8)
    blended = (0.55 * img_uint8 + 0.45 * heat_rgb).astype(np.uint8)
    return blended


def sample_test_images(class_to_idx, n_per_class, seed=SEED):
    df = pd.read_csv(SPLITS_DIR / "test.csv")
    rng = np.random.RandomState(seed)
    samples = []
    for class_name, group in df.groupby("class_name"):
        n = min(n_per_class, len(group))
        chosen = group.sample(n=n, random_state=seed)
        samples.append(chosen)
    return pd.concat(samples).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "vit"], required=True)
    parser.add_argument("--n_per_class", type=int, default=N_EXPLAIN_SAMPLES_PER_CLASS)
    parser.add_argument("--lime_samples", type=int, default=250)
    parser.add_argument("--n_visual_examples", type=int, default=8)
    args = parser.parse_args()

    # Grad-CAM's backward pass hits a known MPS non-contiguous-tensor bug on
    # Apple Silicon (torch.view() during the backward of the hooked conv/attn
    # layer). Explanations only run on a small sampled subset (not the full
    # training loop), so we deliberately use CPU here for stability rather
    # than debugging PyTorch's MPS backward implementation.
    device = torch.device("cpu")
    class_to_idx = json.load(open(SPLITS_DIR / "class_to_idx.json"))
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    ckpt = torch.load(CHECKPOINTS_DIR / f"{args.model}_best.pt", map_location=device)
    model = build_model(args.model, len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    cam = build_gradcam(model, args.model)
    lime_explainer = lime_image.LimeImageExplainer()
    lime_predict_fn = make_lime_predict_fn(model, device)

    samples = sample_test_images(class_to_idx, args.n_per_class)
    print(f"Explaining {len(samples)} sampled test images for model='{args.model}' "
          f"({args.n_per_class}/class) ...")

    rows = []
    all_viz_examples = []  # every sample's visualisation data; a diverse subset is picked after the loop

    for i, row in samples.iterrows():
        filepath, class_name = row["filepath"], row["class_name"]
        true_label = class_to_idx[class_name]
        img = load_and_resize(filepath)

        x = to_model_input(img[None], device)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
        pred_label = int(probs.argmax())
        pred_conf = float(probs[pred_label])
        correct = pred_label == true_label

        gc_map = gradcam_heatmap(cam, img, pred_label, device)
        lm_map, lime_image_and_mask, lm_label = lime_heatmap(lime_explainer, lime_predict_fn, img,
                                                              pred_label, num_samples=args.lime_samples)

        mask = find_segmented_mask(filepath, class_name)
        gc_fg = overlap_score(gc_map, mask) if mask is not None else None
        lm_fg = overlap_score(lm_map, mask) if mask is not None else None

        rows.append({
            "filepath": filepath,
            "class_name": class_name,
            "true_label": true_label,
            "pred_label": pred_label,
            "pred_class": idx_to_class[pred_label],
            "pred_confidence": pred_conf,
            "correct": correct,
            "has_mask": mask is not None,
            "gradcam_foreground_fraction": gc_fg,
            "gradcam_background_fraction": (1 - gc_fg) if gc_fg is not None else None,
            "gradcam_misleading": (1 - gc_fg) > BACKGROUND_FOCUS_THRESHOLD if gc_fg is not None else None,
            "lime_foreground_fraction": lm_fg,
            "lime_background_fraction": (1 - lm_fg) if lm_fg is not None else None,
            "lime_misleading": (1 - lm_fg) > BACKGROUND_FOCUS_THRESHOLD if lm_fg is not None else None,
        })

        all_viz_examples.append((class_name, correct, img, class_name, idx_to_class[pred_label], correct,
                                  gc_map, lime_image_and_mask, mask))

        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(samples)} done")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(METRICS_DIR / f"{args.model}_explanation_quality.csv", index=False)

    summary = {
        "model": args.model,
        "n_samples": len(df_out),
        "n_with_mask": int(df_out["has_mask"].sum()),
        "gradcam_mean_foreground_fraction": float(df_out["gradcam_foreground_fraction"].dropna().mean()),
        "lime_mean_foreground_fraction": float(df_out["lime_foreground_fraction"].dropna().mean()),
        "gradcam_pct_misleading": float(df_out["gradcam_misleading"].dropna().mean() * 100),
        "lime_pct_misleading": float(df_out["lime_misleading"].dropna().mean() * 100),
        "background_focus_threshold": BACKGROUND_FOCUS_THRESHOLD,
    }
    with open(METRICS_DIR / f"{args.model}_explanation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nExplanation quality summary:", json.dumps(summary, indent=2))

    visual_examples = select_diverse_examples(all_viz_examples, args.n_visual_examples)
    plot_visual_grid(visual_examples, args.model)


def select_diverse_examples(all_viz_examples, n):
    """Picks a visually informative subset for the figure: misclassified
    examples first (the most interesting case for an XAI write-up), then
    correct examples spread across distinct classes rather than the first
    N encountered (which would otherwise all come from one class, since
    samples are grouped by class)."""
    incorrect = [e for e in all_viz_examples if not e[1]]
    correct = [e for e in all_viz_examples if e[1]]

    # Track chosen items by id() rather than equality — entries contain numpy
    # arrays, whose `==`/`in` comparison is ambiguous (elementwise), not a
    # simple bool.
    chosen, chosen_ids, seen_classes = [], set(), set()
    for e in incorrect:
        if len(chosen) >= n:
            break
        chosen.append(e)
        chosen_ids.add(id(e))
        seen_classes.add(e[0])
    for e in correct:
        if len(chosen) >= n:
            break
        if e[0] in seen_classes:
            continue
        chosen.append(e)
        chosen_ids.add(id(e))
        seen_classes.add(e[0])
    for e in correct:  # fill any remaining slots if fewer distinct classes than n
        if len(chosen) >= n:
            break
        if id(e) not in chosen_ids:
            chosen.append(e)
            chosen_ids.add(id(e))
    return [e[2:] for e in chosen]


def plot_visual_grid(examples, model_name):
    from viz_style import apply_style
    apply_style()
    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.1 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Original", "Grad-CAM", "LIME (positive regions)", "Segmentation mask"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for r, (img, true_c, pred_c, correct, gc_map, lime_image_and_mask, mask) in enumerate(examples):
        lime_img, lime_mask = lime_image_and_mask
        axes[r, 0].imshow(img)
        status = "correct" if correct else "WRONG"
        color = "#009E73" if correct else "#D55E00"
        axes[r, 0].set_ylabel(f"true: {true_c}\npred: {pred_c} ({status})",
                               fontsize=6.5, color=color, rotation=0, ha="right", va="center", labelpad=60)

        axes[r, 1].imshow(overlay_image(img, gc_map))
        axes[r, 2].imshow(mark_boundaries(lime_img / 255.0 if lime_img.max() > 1 else lime_img, lime_mask))
        if mask is not None:
            axes[r, 3].imshow(img)
            axes[r, 3].imshow(mask, cmap="Greens", alpha=0.4)
        else:
            axes[r, 3].text(0.5, 0.5, "no mask\navailable", ha="center", va="center", fontsize=8)
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)

    fig.suptitle(f"Grad-CAM & LIME explanations — {model_name.upper()}", fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(EXPLANATIONS_DIR / f"{model_name}_explanation_grid.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
