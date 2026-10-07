"""Evaluate a trained checkpoint on a held-out split.

Usage:
    python src/evaluate.py --config configs/cbis_ddsm.yaml --split test
    python src/evaluate.py --config configs/cbis_ddsm.yaml --split val   # if you have no test split yet
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from tqdm import tqdm

from datasets import get_dataloaders
from model import get_model
from utils import get_device, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Defaults to the best checkpoint from training")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"],
                         help="Which split to evaluate on (use 'val' if you don't have a test split yet)")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Decision threshold for the positive (malignant) class. Tune this on val, never on test.")
    parser.add_argument("--find-best-threshold", action="store_true",
                         help="Ignore --threshold; instead sweep thresholds and report the one maximizing accuracy on THIS split. "
                              "Only meaningful when --split val -- never do this on test.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = get_device()
    splits = get_dataloaders(
        cfg["data_dir"], cfg["image_size"], cfg["train"]["batch_size"], cfg["train"]["num_workers"],
        splits=(args.split,),
    )
    eval_ds, eval_loader = splits[args.split]

    model = get_model(cfg["model"]["arch"], cfg["num_classes"], cfg["model"]["pretrained"]).to(device)
    ckpt_path = args.checkpoint or Path(cfg["output"]["checkpoint_dir"]) / f"{cfg['output']['run_name']}_best.pth"
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in tqdm(eval_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())

    all_labels_arr, all_probs_arr = np.array(all_labels), np.array(all_probs)

    threshold = args.threshold
    if args.find_best_threshold:
        candidates = np.linspace(0.05, 0.95, 181)
        accs = [accuracy_score(all_labels_arr, (all_probs_arr >= t).astype(int)) for t in candidates]
        threshold = candidates[int(np.argmax(accs))]
        print(f"Best threshold on this split (by accuracy): {threshold:.3f} -> accuracy {max(accs):.4f}")

    all_preds = (all_probs_arr >= threshold).astype(int).tolist()

    print(f"Checkpoint: {ckpt_path}")
    print(f"Evaluating on split: {args.split}")
    print(f"{args.split} set size: {len(eval_ds)}")
    print(f"Decision threshold used: {threshold:.3f}")
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=cfg["class_names"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds))
    print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")


if __name__ == "__main__":
    main()
