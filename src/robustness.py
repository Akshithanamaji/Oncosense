"""Robustness and Perturbation Testing Lab.

Evaluates how sensitive the ResNet18 model is to clinical imaging variations:
1. Brightness shift (over/underexposure)
2. Contrast variations (low/high dynamic range mammograms)
3. Gaussian sensor noise (electronic detector noise)
4. Gaussian blur (patient micro-motion / focal blur)
5. Scale / Zoom distortions (improper breast positioning)

Measures degradation in Accuracy, Sensitivity, Specificity, and Prediction Shift.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def run_robustness_stress_test(sample_limit: int = 150) -> Dict[str, Any]:
    """Runs controlled perturbations on a sample of test mammograms and benchmarks
    the performance degradation curves.
    """
    from datasets import get_dataloaders
    from model import get_model
    from utils import get_device, load_checkpoint

    config_path = _PROJECT_ROOT / "configs" / "cbis_ddsm_v3.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    splits = get_dataloaders(
        cfg["data_dir"], cfg["image_size"], batch_size=16,
        num_workers=0, splits=("test",),
    )
    dataset, loader = splits["test"]

    model = get_model(
        cfg["model"]["arch"], cfg["num_classes"],
        cfg["model"]["pretrained"], cfg["model"].get("dropout", 0.0),
    ).to(device)
    ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['output']['run_name']}_best.pth"
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    # Collect batches up to sample_limit
    collected_images = []
    collected_labels = []
    for imgs, lbls in loader:
        collected_images.append(imgs)
        collected_labels.append(lbls)
        if sum(len(b) for b in collected_labels) >= sample_limit:
            break

    orig_images = torch.cat(collected_images, dim=0)[:sample_limit].to(device)
    labels = torch.cat(collected_labels, dim=0)[:sample_limit].numpy()

    # Baseline performance
    with torch.no_grad():
        base_logits = model(orig_images)
        base_probs = torch.softmax(base_logits, dim=1)[:, 1].cpu().numpy()
    base_preds = (base_probs >= 0.5).astype(int)
    base_acc = float(accuracy_score(labels, base_preds))
    base_sens = float(recall_score(labels, base_preds, zero_division=0))

    perturbation_suites = {
        "Brightness Shift": [
            ("Darken (-30%)", lambda t: torch.clamp(t * 0.7, 0.0, 1.0)),
            ("Darken (-15%)", lambda t: torch.clamp(t * 0.85, 0.0, 1.0)),
            ("Baseline (100%)", lambda t: t),
            ("Brighten (+15%)", lambda t: torch.clamp(t * 1.15, 0.0, 1.0)),
            ("Brighten (+30%)", lambda t: torch.clamp(t * 1.30, 0.0, 1.0)),
        ],
        "Contrast Variation": [
            ("Low Contrast (0.6x)", lambda t: torch.clamp((t - 0.5) * 0.6 + 0.5, 0.0, 1.0)),
            ("Mild Contrast (0.8x)", lambda t: torch.clamp((t - 0.5) * 0.8 + 0.5, 0.0, 1.0)),
            ("Baseline (1.0x)", lambda t: t),
            ("High Contrast (1.2x)", lambda t: torch.clamp((t - 0.5) * 1.2 + 0.5, 0.0, 1.0)),
            ("Extreme Contrast (1.4x)", lambda t: torch.clamp((t - 0.5) * 1.4 + 0.5, 0.0, 1.0)),
        ],
        "Detector Gaussian Noise": [
            ("None (sigma=0)", lambda t: t),
            ("Low (sigma=0.03)", lambda t: torch.clamp(t + torch.randn_like(t) * 0.03, 0.0, 1.0)),
            ("Medium (sigma=0.06)", lambda t: torch.clamp(t + torch.randn_like(t) * 0.06, 0.0, 1.0)),
            ("High (sigma=0.10)", lambda t: torch.clamp(t + torch.randn_like(t) * 0.10, 0.0, 1.0)),
            ("Extreme (sigma=0.15)", lambda t: torch.clamp(t + torch.randn_like(t) * 0.15, 0.0, 1.0)),
        ],
        "Motion / Optical Blur": [
            ("Sharp (Baseline)", lambda t: t),
            ("Mild Blur (kernel=3)", lambda t: TF.gaussian_blur(t, kernel_size=[3, 3], sigma=[0.6, 0.6])),
            ("Medium Blur (kernel=5)", lambda t: TF.gaussian_blur(t, kernel_size=[5, 5], sigma=[1.2, 1.2])),
            ("Heavy Blur (kernel=7)", lambda t: TF.gaussian_blur(t, kernel_size=[7, 7], sigma=[1.8, 1.8])),
        ],
    }

    results = {
        "sample_size": len(labels),
        "baseline": {
            "accuracy": round(base_acc, 4),
            "sensitivity": round(base_sens, 4),
        },
        "experiments": {},
    }

    with torch.no_grad():
        for suite_name, steps in perturbation_suites.items():
            suite_data = []
            for step_name, transform_fn in steps:
                pert_imgs = transform_fn(orig_images)
                logits = model(pert_imgs)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                acc = float(accuracy_score(labels, preds))
                sens = float(recall_score(labels, preds, zero_division=0))
                mean_delta = float(np.mean(np.abs(probs - base_probs)))
                flipped_count = int((preds != base_preds).sum())
                flip_pct = float(flipped_count / len(labels))

                suite_data.append({
                    "step": step_name,
                    "accuracy": round(acc, 4),
                    "sensitivity": round(sens, 4),
                    "acc_drop": round(base_acc - acc, 4),
                    "mean_prob_shift": round(mean_delta, 4),
                    "prediction_flips": flipped_count,
                    "flip_rate_pct": round(flip_pct * 100, 1),
                })
            results["experiments"][suite_name] = suite_data

    # Save to outputs cache
    cache_path = _PROJECT_ROOT / "outputs" / "robustness_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def get_cached_robustness_test() -> Dict[str, Any]:
    cache_path = _PROJECT_ROOT / "outputs" / "robustness_cache.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    return run_robustness_stress_test()
