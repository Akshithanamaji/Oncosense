"""Grad-CAM visualization: shows which regions of a patch/slide drove the
model's prediction. Useful for the report/viva to argue the model is looking
at the tissue, not artifacts.

Usage:
    python src/gradcam.py --config configs/idc.yaml --num-samples 8
    python src/gradcam.py --config configs/idc.yaml --image path/to/one_patch.png
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from datasets import build_transforms, get_dataloaders
from model import get_model
from utils import get_device, load_checkpoint


def get_target_layer(model, arch: str):
    if arch == "simple_cnn":
        return model.features[-3]  # last Conv2d in the block before pooling
    if arch == "resnet18":
        return model.layer4[-1]
    if arch == "efficientnet_b0":
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer defined for arch: {arch}")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor, class_idx=1):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = torch.softmax(output, dim=1)[0].detach().cpu().numpy()
        return cam, probs


def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray) -> np.ndarray:
    img = np.array(pil_image.convert("RGB"))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.6 * img + 0.4 * heatmap).astype(np.uint8)
    return np.concatenate([img, overlay], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image", default=None, help="Single image to visualize (skips test-set sampling)")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--out-dir", default="outputs/gradcam")
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
            sample_dir = Path(cfg["data_dir"]) / "val"  # no test split prepared yet
        all_images = list(sample_dir.rglob("*.png")) + list(sample_dir.rglob("*.jpg"))
        image_paths = random.sample(all_images, min(args.num_samples, len(all_images)))

    for img_path in image_paths:
        pil_img = Image.open(img_path)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        cam, probs = cam_engine(input_tensor, class_idx=1)
        result = overlay_heatmap(pil_img.resize((cfg["image_size"], cfg["image_size"])), cam)

        out_path = out_dir / f"{img_path.stem}_p{probs[1]:.2f}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"{img_path.name}: P(positive/malignant)={probs[1]:.3f} -> saved {out_path}")


if __name__ == "__main__":
    main()
