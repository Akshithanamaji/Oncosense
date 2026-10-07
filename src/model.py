import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Small CNN for 50x50 IDC patches. Trains fast from scratch, no pretrained weights needed."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def get_model(arch: str, num_classes: int, pretrained: bool = False, dropout: float = 0.0):
    if arch == "simple_cnn":
        return SimpleCNN(num_classes)

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        return model

    if arch == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout or 0.2), nn.Linear(in_features, num_classes))
        return model

    raise ValueError(f"Unknown arch: {arch}")


def get_classifier_module(model, arch: str):
    if arch == "resnet18":
        return model.fc
    if arch == "efficientnet_b0":
        return model.classifier
    raise ValueError(f"Unknown arch: {arch}")


def get_param_groups(model, arch: str, base_lr: float, backbone_lr_mult: float = 0.1):
    """Discriminative LR groups: classifier head at base_lr, backbone at a fraction of it.

    Combine with set_backbone_requires_grad(..., False) for the first few epochs
    to warm up the (randomly-initialized) head before touching pretrained weights --
    this stabilizes fine-tuning on small datasets.
    """
    head_params = list(get_classifier_module(model, arch).parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    return [
        {"params": backbone_params, "lr": base_lr * backbone_lr_mult},
        {"params": head_params, "lr": base_lr},
    ]


def set_backbone_requires_grad(model, arch: str, requires_grad: bool):
    head_ids = {id(p) for p in get_classifier_module(model, arch).parameters()}
    for p in model.parameters():
        if id(p) not in head_ids:
            p.requires_grad = requires_grad
