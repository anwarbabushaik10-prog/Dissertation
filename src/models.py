"""
Model factory for the two prototype architectures compared in the
dissertation: a transfer-learning CNN (MobileNetV2) and a lightweight
Vision Transformer (ViT-Tiny), both initialised from ImageNet weights via
`timm`. Keeping both behind one `build_model()` function keeps train.py,
evaluate.py and explain.py architecture-agnostic.
"""
import timm
import torch.nn as nn

from config import CNN_BACKBONE, VIT_BACKBONE, IMAGE_SIZE, DROPOUT


def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    model_name: "cnn" or "vit"
    """
    if model_name == "cnn":
        model = timm.create_model(
            CNN_BACKBONE, pretrained=pretrained, num_classes=num_classes,
            drop_rate=DROPOUT,
        )
    elif model_name == "vit":
        model = timm.create_model(
            VIT_BACKBONE, pretrained=pretrained, num_classes=num_classes,
            img_size=IMAGE_SIZE, drop_rate=DROPOUT,
        )
    else:
        raise ValueError(f"Unknown model_name '{model_name}', expected 'cnn' or 'vit'")
    return model


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    """
    Freezes/unfreezes everything except the final classifier head, so that
    the first `FREEZE_BACKBONE_EPOCHS` only train the newly-initialised head
    (standard transfer-learning warm-up) before fine-tuning the backbone at
    a lower learning rate.
    """
    head_names = ("classifier", "fc", "head")
    for name, param in model.named_parameters():
        is_head = any(name.startswith(h) for h in head_names)
        if not is_head:
            param.requires_grad = trainable


def param_groups(model: nn.Module, lr_head: float, lr_backbone: float):
    head_names = ("classifier", "fc", "head")
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if any(name.startswith(h) for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ]
