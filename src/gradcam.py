"""Grad-CAM visualization & Explainable AI engine for breast cancer screening.

Shows which regions of a mammogram influenced ResNet18's prediction, and provides
scientific tools to evaluate whether highlighted areas correspond to annotated
parenchymal lesions versus background/scanner artifacts.

Research principle:
    A heatmap alone does not prove the model is correct. A neural network may
    achieve high confidence or correct labels by keying off background markers,
    chest wall borders, or scanner artifacts (shortcut learning). Genuine clinical
    validity requires verifying whether highlighted activations correspond to
    radiologically annotated lesions.

Usage:
    python src/gradcam.py --config configs/cbis_ddsm_v3.yaml --num-samples 8
    python src/gradcam.py --config configs/cbis_ddsm_v3.yaml --image path/to/mammogram.png
    python src/gradcam.py --config configs/cbis_ddsm_v3.yaml --evaluate-correspondence
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
_ROOT_DIR = _THIS_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

try:
    from datasets import build_transforms, get_dataloaders
    from model import get_model
    from utils import get_device, load_checkpoint
except ImportError:
    from src.datasets import build_transforms, get_dataloaders
    from src.model import get_model
    from src.utils import get_device, load_checkpoint


COLORMAP_DICT = {
    "jet": cv2.COLORMAP_JET,
    "turbo": cv2.COLORMAP_TURBO,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "inferno": cv2.COLORMAP_INFERNO,
    "magma": cv2.COLORMAP_MAGMA,
}


def get_target_layer(model, arch: str):
    """Returns the target convolutional layer for Grad-CAM for the given architecture.

    For ResNet18, layer4[-1] is the final residual BasicBlock, whose output maps
    capture the highest-level spatial representations before global pooling.
    """
    if arch == "simple_cnn":
        return model.features[-3]  # last Conv2d in the block before pooling
    if arch == "resnet18":
        return model.layer4[-1]
    if arch == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer defined for arch: {arch}")


class GradCAM:
    """Computes Grad-CAM (Gradient-weighted Class Activation Mapping).

    Attributes:
        model: PyTorch model (e.g. ResNet18).
        target_layer: The target convolutional layer (e.g. model.layer4[-1]).
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        self._hooks.append(self.target_layer.register_forward_hook(self._save_activations))
        self._hooks.append(self.target_layer.register_full_backward_hook(self._save_gradients))

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(self, input_tensor, class_idx=None, retain_graph=False):
        """Generates Grad-CAM heatmap for input_tensor.

        Args:
            input_tensor: 4D tensor (1, C, H, W).
            class_idx: int or None. If None, targets the model's top predicted class
                       to show which regions influenced ResNet18's actual prediction.
            retain_graph: whether to retain computation graph.

        Returns:
            cam: 2D numpy array [0.0, 1.0] of shape (H, W).
            probs: 1D numpy array of softmax class probabilities.
        """
        with torch.enable_grad():
            output = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(output.argmax(dim=1).item())

            self.model.zero_grad()
            score = output[0, class_idx]
            score.backward(retain_graph=retain_graph)

        # Global average pooling of gradients: alpha_k^c
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        # ReLU of linear combination: L_GradCAM^c = ReLU(sum_k alpha_k^c * A^k)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        # Upsample to input resolution
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        denom = cam.max() - cam.min()
        if denom > 1e-8:
            cam = (cam - cam.min()) / denom
        else:
            cam = np.zeros_like(cam)

        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, probs

    def generate_all_classes(self, input_tensor):
        """Computes Grad-CAM for all classes (e.g. Benign vs Malignant) in one pass.

        Returns:
            cams: dict {class_idx: cam_array}
            probs: softmax probability array
            pred_idx: predicted class index
        """
        with torch.enable_grad():
            output = self.model(input_tensor)
            pred_idx = int(output.argmax(dim=1).item())
            probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
            cams = {}
            num_classes = output.shape[1]
            for c in range(num_classes):
                self.model.zero_grad()
                output[0, c].backward(retain_graph=(c < num_classes - 1))
                weights = self.gradients.mean(dim=(2, 3), keepdim=True)
                cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
                cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
                cam = cam.squeeze().cpu().numpy()
                denom = cam.max() - cam.min()
                if denom > 1e-8:
                    cam = (cam - cam.min()) / denom
                else:
                    cam = np.zeros_like(cam)
                cams[c] = cam
        return cams, probs, pred_idx


def extract_breast_tissue_mask(image: Image.Image | np.ndarray) -> np.ndarray:
    """Extracts a binary mask of the breast parenchyma using Otsu thresholding

    and morphological opening, separating tissue from dark background, margins,
    and scanner text markers.
    """
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("L"), dtype=np.uint8)
    elif isinstance(image, np.ndarray) and image.ndim == 3:
        arr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        arr = np.asarray(image, dtype=np.uint8)

    _, mask = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.uint8(labels == largest_label) * 255
    return mask


def compute_saliency_metrics(cam: np.ndarray, threshold: float = 0.5) -> dict:
    """Calculates spatial and distribution metrics for a Grad-CAM heatmap."""
    h, w = cam.shape
    peak_y, peak_x = np.unravel_index(np.argmax(cam), cam.shape)
    peak_value = float(cam[peak_y, peak_x])
    mean_val = float(cam.mean())

    salient_mask = (cam >= threshold).astype(np.uint8) * 255
    salient_pixels = int(np.sum(salient_mask > 0))
    salient_area_pct = float((salient_pixels / (h * w)) * 100.0)
    focal_ratio = float(peak_value / (mean_val + 1e-6))

    contours, _ = cv2.findContours(salient_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(largest_cnt)
        bbox = (int(bx), int(by), int(bw), int(bh))
    else:
        margin = max(10, min(h, w) // 10)
        bx = max(0, peak_x - margin)
        by = max(0, peak_y - margin)
        bw = min(w - bx, margin * 2)
        bh = min(h - by, margin * 2)
        bbox = (int(bx), int(by), int(bw), int(bh))

    return {
        "peak_x": int(peak_x),
        "peak_y": int(peak_y),
        "peak_value": round(peak_value, 4),
        "mean_activation": round(mean_val, 4),
        "salient_area_pct": round(salient_area_pct, 2),
        "focal_ratio": round(focal_ratio, 2),
        "bbox": bbox,
        "is_focal": focal_ratio > 3.0 and salient_area_pct < 18.0,
    }


def evaluate_lesion_correspondence(
    cam: np.ndarray,
    image: Image.Image | np.ndarray,
    ground_truth_bbox: tuple | None = None,
    ground_truth_mask: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict:
    """Evaluates whether Grad-CAM highlighted areas correspond to annotated lesions

    or tissue structures versus background artifacts.

    Research principle:
        A heatmap alone does not prove the model is correct. A high classification
        confidence or correct prediction may be driven by confounding artifacts
        (borders, scanner tags, chest wall). This function computes quantitative
        tissue alignment, pointing game hit, overlap (IoU/Dice), and artifact risk.
    """
    tissue_mask = extract_breast_tissue_mask(image)
    if tissue_mask.shape != cam.shape:
        tissue_mask = cv2.resize(tissue_mask, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST)

    cam_energy = float(cam.sum()) + 1e-8
    inside_tissue_energy = float((cam * (tissue_mask > 0)).sum())
    tissue_alignment_pct = round((inside_tissue_energy / cam_energy) * 100.0, 1)

    saliency = compute_saliency_metrics(cam, threshold=threshold)
    peak_x, peak_y = saliency["peak_x"], saliency["peak_y"]
    peak_inside_tissue = bool(tissue_mask[peak_y, peak_x] > 0)

    pointing_hit = None
    iou = None
    dice = None
    energy_in_lesion = None

    if ground_truth_mask is not None:
        gt_mask = ground_truth_mask
        if gt_mask.shape != cam.shape:
            gt_mask = cv2.resize(gt_mask, (cam.shape[1], cam.shape[0]), interpolation=cv2.INTER_NEAREST)
        pointing_hit = bool(gt_mask[peak_y, peak_x] > 0)
        salient_mask = (cam >= threshold)
        gt_bool = gt_mask > 0
        intersection = np.logical_and(salient_mask, gt_bool).sum()
        union = np.logical_or(salient_mask, gt_bool).sum()
        iou = round(float(intersection / (union + 1e-8)), 4)
        dice = round(float((2.0 * intersection) / (salient_mask.sum() + gt_bool.sum() + 1e-8)), 4)
        energy_in_lesion = round(float((cam * gt_bool).sum() / cam_energy) * 100.0, 1)

    elif ground_truth_bbox is not None:
        gx, gy, gw, gh = ground_truth_bbox
        pointing_hit = bool(gx <= peak_x <= gx + gw and gy <= peak_y <= gy + gh)
        gt_mask = np.zeros_like(tissue_mask)
        gt_mask[gy : gy + gh, gx : gx + gw] = 255
        gt_bool = gt_mask > 0
        salient_mask = (cam >= threshold)
        intersection = np.logical_and(salient_mask, gt_bool).sum()
        union = np.logical_or(salient_mask, gt_bool).sum()
        iou = round(float(intersection / (union + 1e-8)), 4)
        dice = round(float((2.0 * intersection) / (salient_mask.sum() + gt_bool.sum() + 1e-8)), 4)
        energy_in_lesion = round(float((cam * gt_bool).sum() / cam_energy) * 100.0, 1)

    # Artifact Risk Assessment
    if tissue_alignment_pct >= 85.0 and peak_inside_tissue:
        artifact_risk = "Low Risk (Strong Tissue Alignment)"
        artifact_verdict = "Parenchymal Alignment Confirmed"
        research_note = (
            "Peak attention is firmly situated within breast parenchymal tissue, "
            "supporting biological plausibility rather than shortcut learning."
        )
    elif tissue_alignment_pct >= 65.0:
        artifact_risk = "Moderate Risk (Border / Margin Influence)"
        artifact_verdict = "Mixed Parenchymal / Margin Attention"
        research_note = (
            "Model attends partially to tissue, but significant activation extends toward "
            "pectoral or border margins. Cross-reference with clinical presentation."
        )
    else:
        artifact_risk = "High Risk (Potential Artifact Shortcut)"
        artifact_verdict = "Non-Parenchymal Artifact Risk"
        research_note = (
            "A high fraction of attention falls outside breast parenchyma. "
            "Caution: A heatmap alone does not prove the model is correct. The model may be keying off artifacts."
        )

    return {
        "tissue_alignment_pct": tissue_alignment_pct,
        "peak_inside_tissue": peak_inside_tissue,
        "pointing_hit": pointing_hit,
        "iou": iou,
        "dice": dice,
        "energy_in_lesion": energy_in_lesion,
        "artifact_risk": artifact_risk,
        "artifact_verdict": artifact_verdict,
        "research_note": research_note,
        "saliency": saliency,
    }


def overlay_heatmap(
    pil_image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
    threshold: float = 0.0,
    show_bbox: bool = False,
    show_centroid: bool = False,
    mode: str = "concat",
) -> np.ndarray:
    """Overlays Grad-CAM heatmap over mammogram image.

    Args:
        pil_image: PIL RGB image.
        cam: 2D numpy array [0.0, 1.0].
        alpha: Blending weight for heatmap (0.0: mammogram only, 1.0: heatmap only).
        colormap: 'jet', 'turbo', 'viridis', 'inferno', 'magma'.
        threshold: Saliency cutoff below which activation is suppressed.
        show_bbox: If True, draws a bounding box around the primary salient cluster.
        show_centroid: If True, draws a crosshair at the peak activation coordinate.
        mode: 'concat' (returns [orig, overlay]), 'overlay_only' (returns overlay),
              or 'both' (returns tuple of (orig, overlay)).

    Returns:
        np.ndarray (uint8 RGB)
    """
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    if cam.shape != (h, w):
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)

    processed_cam = np.copy(cam)
    # Mask of areas that exceed the noise cutoff threshold
    active_mask = processed_cam >= threshold if threshold > 0.0 else processed_cam > 0.01

    if threshold > 0.0:
        processed_cam[~active_mask] = 0.0
        max_v = processed_cam.max()
        if max_v > 1e-6:
            processed_cam = processed_cam / max_v

    cv_cmap = COLORMAP_DICT.get(colormap.lower(), cv2.COLORMAP_JET)
    heatmap = cv2.applyColorMap(np.uint8(255 * processed_cam), cv_cmap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Per-pixel alpha blending: Only blend heatmap over regions where activation exceeds threshold.
    # Areas below threshold keep the clean original mammogram image without tint or haze.
    effective_alpha = np.zeros((h, w, 1), dtype=np.float32)
    effective_alpha[active_mask] = alpha * processed_cam[active_mask, np.newaxis]
    # Bound between 0.2*alpha and alpha so the salient region remains clearly visible
    effective_alpha[active_mask] = np.clip(effective_alpha[active_mask], 0.25 * alpha, alpha)

    img_f = img.astype(np.float32)
    heatmap_f = heatmap.astype(np.float32)
    overlay = (img_f * (1.0 - effective_alpha) + heatmap_f * effective_alpha).astype(np.uint8)

    if show_bbox or show_centroid:
        saliency = compute_saliency_metrics(cam, threshold=max(0.4, threshold))
        if show_bbox and saliency.get("bbox"):
            bx, by, bw, bh = saliency["bbox"]
            cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (255, 60, 120), 2)
            cv2.putText(
                overlay, "Peak Attention", (bx, max(14, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 60, 120), 1, cv2.LINE_AA,
            )
        if show_centroid:
            px, py = saliency["peak_x"], saliency["peak_y"]
            cv2.drawMarker(overlay, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)

    if mode == "overlay_only":
        return overlay
    if mode == "both":
        return img, overlay
    return np.concatenate([img, overlay], axis=1)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM explainability evaluation for breast cancer detection")
    parser.add_argument("--config", required=True, help="YAML config file path")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--image", default=None, help="Single image to visualize (skips test-set sampling)")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of samples to visualize")
    parser.add_argument("--target-class", choices=["auto", "malignant", "benign", "all"], default="auto",
                        help="Class to explain ('auto' explains the model's actual prediction)")
    parser.add_argument("--colormap", default="turbo", choices=["turbo", "jet", "viridis", "inferno", "magma"])
    parser.add_argument("--alpha", type=float, default=0.45, help="Heatmap opacity")
    parser.add_argument("--out-dir", default="outputs/gradcam", help="Output directory")
    parser.add_argument("--evaluate-correspondence", action="store_true",
                        help="Compute tissue alignment & lesion correspondence metrics")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    model = get_model(cfg["model"]["arch"], cfg["num_classes"], cfg["model"]["pretrained"]).to(device)
    ckpt_path = args.checkpoint or Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['output']['run_name']}_best.pth"
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    target_layer = get_target_layer(model, cfg["model"]["arch"])
    cam_engine = GradCAM(model, target_layer)
    transform = build_transforms(cfg["image_size"], train=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        image_paths = [Path(args.image)]
    else:
        sample_dir = Path(cfg["data_dir"]) / "test"
        if not sample_dir.exists() or not any(sample_dir.rglob("*.png")):
            sample_dir = Path(cfg["data_dir"]) / "val"
        all_images = list(sample_dir.rglob("*.png")) + list(sample_dir.rglob("*.jpg"))
        image_paths = random.sample(all_images, min(args.num_samples, len(all_images)))

    class_names = cfg.get("class_names", ["benign", "malignant"])
    summary_records = []

    print(f"\nRunning Grad-CAM on {len(image_paths)} images using {cfg['model']['arch']} (device: {device})...")
    print("=" * 78)

    for img_path in image_paths:
        pil_img = Image.open(img_path).convert("RGB")
        resized_img = pil_img.resize((cfg["image_size"], cfg["image_size"]))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        if args.target_class == "auto":
            cam, probs = cam_engine(input_tensor, class_idx=None)
            pred_idx = int(np.argmax(probs))
            explained_class = class_names[pred_idx]
            target_idx = pred_idx
        elif args.target_class == "malignant":
            cam, probs = cam_engine(input_tensor, class_idx=1)
            pred_idx = int(np.argmax(probs))
            explained_class = "malignant"
            target_idx = 1
        elif args.target_class == "benign":
            cam, probs = cam_engine(input_tensor, class_idx=0)
            pred_idx = int(np.argmax(probs))
            explained_class = "benign"
            target_idx = 0
        else:  # all
            cams, probs, pred_idx = cam_engine.generate_all_classes(input_tensor)
            cam = cams[pred_idx]
            explained_class = f"auto_{class_names[pred_idx]}"
            target_idx = pred_idx

        # Generate overlay
        result = overlay_heatmap(
            resized_img, cam, alpha=args.alpha, colormap=args.colormap,
            show_bbox=True, show_centroid=True, mode="concat",
        )

        out_path = out_dir / f"{img_path.stem}_pred_{class_names[pred_idx]}_pMal{probs[1]:.2f}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        # Evaluate correspondence
        eval_metrics = evaluate_lesion_correspondence(cam, resized_img, threshold=0.5)
        tissue_pct = eval_metrics["tissue_alignment_pct"]
        artifact_risk = eval_metrics["artifact_risk"]

        summary_records.append({
            "image": img_path.name,
            "pred_class": class_names[pred_idx],
            "p_benign": round(float(probs[0]), 4),
            "p_malignant": round(float(probs[1]), 4),
            "explained_target": explained_class,
            "tissue_alignment_pct": tissue_pct,
            "peak_inside_tissue": eval_metrics["peak_inside_tissue"],
            "artifact_risk": artifact_risk,
            "saved_overlay": str(out_path),
        })

        print(f"[{img_path.name}] Pred: {class_names[pred_idx]} (P(mal)={probs[1]:.3f}) | "
              f"Tissue Alignment: {tissue_pct}% | Risk: {artifact_risk} -> Saved {out_path.name}")

    # Save summary CSV
    summary_csv = out_dir / "gradcam_evaluation_summary.csv"
    if summary_records:
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_records[0].keys()))
            writer.writeheader()
            writer.writerows(summary_records)
        print("=" * 78)
        print(f"Evaluation summary written to: {summary_csv}")


if __name__ == "__main__":
    main()
