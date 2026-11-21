"""Unit and integration tests for Explainable AI — Grad-CAM."""
import sys
from pathlib import Path

# Add project root and src to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from src.datasets import build_transforms
from src.gradcam import (
    COLORMAP_DICT,
    GradCAM,
    compute_saliency_metrics,
    evaluate_lesion_correspondence,
    extract_breast_tissue_mask,
    get_target_layer,
    overlay_heatmap,
)
from src.model import get_model
from src.utils import load_checkpoint


def test_gradcam_resnet18():
    print("\n--- Test 1: ResNet18 Architecture & Hooking ---")
    with open(ROOT / "configs" / "cbis_ddsm_v3.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    model = get_model(cfg["model"]["arch"], cfg["num_classes"], pretrained=False).to(device)
    ckpt_path = ROOT / "outputs" / "checkpoints" / "cbis_ddsm_v3_best.pth"
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    target_layer = get_target_layer(model, "resnet18")
    assert target_layer is not None, "Target layer for resnet18 should not be None"

    cam_engine = GradCAM(model, target_layer)

    # Test on synthetic tensor
    dummy_input = torch.randn(1, 3, cfg["image_size"], cfg["image_size"])
    cam_auto, probs = cam_engine(dummy_input, class_idx=None)
    assert cam_auto.shape == (cfg["image_size"], cfg["image_size"])
    assert 0.0 <= cam_auto.min() and cam_auto.max() <= 1.0001
    assert len(probs) == 2
    assert np.isclose(probs.sum(), 1.0, atol=1e-4)
    print("[PASS] Synthetic tensor Grad-CAM computation passed!")

    print("\n--- Test 2: Multi-class Grad-CAM Generation ---")
    cams, probs_all, pred_idx = cam_engine.generate_all_classes(dummy_input)
    assert 0 in cams and 1 in cams
    assert cams[0].shape == (cfg["image_size"], cfg["image_size"])
    assert cams[1].shape == (cfg["image_size"], cfg["image_size"])
    assert pred_idx in [0, 1]
    print(f"[PASS] Generated CAMs for both classes successfully (Pred class: {pred_idx})!")

    print("\n--- Test 3: Real Mammogram Inference & Lesion Correspondence ---")
    test_malignant = list((ROOT / "data" / "processed" / "cbis_ddsm" / "test" / "malignant").glob("*.png"))
    assert len(test_malignant) > 0, "Should find test images"
    sample_img_path = test_malignant[0]
    print(f"Testing real scan: {sample_img_path.name}")

    pil_img = Image.open(sample_img_path).convert("RGB")
    resized_img = pil_img.resize((cfg["image_size"], cfg["image_size"]))
    transform = build_transforms(cfg["image_size"], train=False)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    cams, probs, pred_idx = cam_engine.generate_all_classes(input_tensor)
    cam_pred = cams[pred_idx]
    pred_label = cfg["class_names"][pred_idx]
    print(f"Model prediction: {pred_label} (P(benign)={probs[0]:.3f}, P(mal)={probs[1]:.3f})")

    # Tissue mask
    tissue_mask = extract_breast_tissue_mask(resized_img)
    assert tissue_mask.shape == (cfg["image_size"], cfg["image_size"])
    tissue_pixels = np.sum(tissue_mask > 0)
    assert tissue_pixels > 0, "Tissue mask should detect breast parenchyma"
    print(f"[PASS] Tissue mask segmented {tissue_pixels} parenchyma pixels")

    # Correspondence evaluation
    eval_res = evaluate_lesion_correspondence(cam_pred, resized_img, threshold=0.4)
    assert "tissue_alignment_pct" in eval_res
    assert "artifact_risk" in eval_res
    assert "artifact_verdict" in eval_res
    assert "research_note" in eval_res
    assert "saliency" in eval_res

    tissue_pct = eval_res["tissue_alignment_pct"]
    saliency = eval_res["saliency"]
    print(f"[PASS] Correspondence: {tissue_pct:.1f}% energy inside breast parenchyma")
    print(f"[PASS] Peak focus at ({saliency['peak_x']}, {saliency['peak_y']}) with intensity {saliency['peak_value']:.2f}")
    print(f"[PASS] Risk assessment: {eval_res['artifact_verdict']} ({eval_res['artifact_risk']})")
    print(f"[PASS] Research note: {eval_res['research_note']}")

    print("\n--- Test 4: Overlay & Scientific Colormaps ---")
    for cmap in ["turbo", "jet", "viridis", "inferno"]:
        overlay = overlay_heatmap(
            resized_img,
            cam_pred,
            alpha=0.45,
            colormap=cmap,
            threshold=0.15,
            show_bbox=True,
            show_centroid=True,
            mode="overlay_only",
        )
        assert overlay.shape == (cfg["image_size"], cfg["image_size"], 3)
        assert overlay.dtype == np.uint8

    # Backward compatibility mode
    concat_overlay = overlay_heatmap(resized_img, cam_pred, mode="concat")
    assert concat_overlay.shape == (cfg["image_size"], cfg["image_size"] * 2, 3)
    print("[PASS] All colormaps and overlay modes verified successfully!")

    print("\n=======================================================")
    print("ALL TESTS PASSED! Explainable AI Grad-CAM fully verified.")
    print("=======================================================")


if __name__ == "__main__":
    test_gradcam_resnet18()
