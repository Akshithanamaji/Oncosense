"""Evaluation engine — runs a trained checkpoint on a held-out split and
caches every metric the Research Dashboard needs.

The heavy work (model inference on every test image) runs once; results are
written to ``outputs/evaluation_cache.json`` so the Streamlit page can load
them instantly on every subsequent visit.

Usage (standalone):
    python src/evaluation_engine.py --config configs/cbis_ddsm_v3.yaml

From Streamlit (imported):
    from evaluation_engine import run_full_evaluation, load_cached_evaluation
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ---------------------------------------------------------------------------
# Import helpers — allow running both as `python src/evaluation_engine.py`
# and as an import from Streamlit (where src/ may not be on sys.path yet).
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from datasets import get_dataloaders  # noqa: E402
from model import get_model  # noqa: E402
from utils import get_device, load_checkpoint  # noqa: E402

# Where the JSON cache lives — always relative to project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_PATH = _PROJECT_ROOT / "outputs" / "evaluation_cache.json"


def _compute_threshold_curve(labels: np.ndarray, probs: np.ndarray,
                              num_points: int = 91) -> list[dict]:
    """Sweep decision thresholds and record accuracy / F1 / sensitivity /
    specificity at each one.  Used by the dashboard's Threshold Analysis tab.
    """
    thresholds = np.linspace(0.05, 0.95, num_points)
    curve = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        curve.append({
            "threshold": round(float(t), 4),
            "accuracy": round(float(accuracy_score(labels, preds)), 4),
            "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
            "sensitivity": round(float(tp / max(tp + fn, 1)), 4),
            "specificity": round(float(tn / max(tn + fp, 1)), 4),
            "precision": round(float(precision_score(labels, preds, zero_division=0)), 4),
        })
    return curve


def run_full_evaluation(config_path: str, split: str = "test",
                         threshold: float = 0.5,
                         progress_callback=None) -> dict:
    """Run model inference on *split*, compute all metrics, and return a dict
    that is JSON-serialisable.  Optionally calls ``progress_callback(done, total)``
    after each batch so a Streamlit progress bar can update.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = get_device()

    # ---- Load data -------------------------------------------------------
    splits = get_dataloaders(
        cfg["data_dir"], cfg["image_size"], cfg["train"]["batch_size"],
        cfg["train"]["num_workers"], splits=(split,),
    )
    dataset, loader = splits[split]
    class_names = cfg.get("class_names", ["benign", "malignant"])
    num_images = len(dataset)

    # ---- Load model ------------------------------------------------------
    model = get_model(
        cfg["model"]["arch"], cfg["num_classes"],
        cfg["model"]["pretrained"], cfg["model"].get("dropout", 0.0),
    ).to(device)
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / f"{cfg['output']['run_name']}_best.pth"
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    # ---- Inference -------------------------------------------------------
    all_labels: list[int] = []
    all_probs: list[float] = []
    batch_times: list[float] = []
    done = 0

    with torch.no_grad():
        for images, labels in loader:
            t0 = time.perf_counter()
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu()
            elapsed = time.perf_counter() - t0

            all_probs.extend(probs.tolist())
            all_labels.extend(labels.tolist())
            batch_times.append(elapsed / len(labels))  # per-image

            done += len(labels)
            if progress_callback:
                progress_callback(done, num_images)

    labels_arr = np.array(all_labels)
    probs_arr = np.array(all_probs)
    preds_arr = (probs_arr >= threshold).astype(int)

    # ---- Scalar metrics --------------------------------------------------
    tn, fp, fn, tp = confusion_matrix(labels_arr, preds_arr, labels=[0, 1]).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    roc_auc = float(roc_auc_score(labels_arr, probs_arr))

    # ---- ROC curve -------------------------------------------------------
    fpr, tpr, roc_thresholds = roc_curve(labels_arr, probs_arr)
    # Downsample for JSON size (keep at most 200 points)
    if len(fpr) > 200:
        idx = np.linspace(0, len(fpr) - 1, 200, dtype=int)
        fpr, tpr, roc_thresholds = fpr[idx], tpr[idx], roc_thresholds[idx]

    # ---- Threshold sweep -------------------------------------------------
    threshold_curve = _compute_threshold_curve(labels_arr, probs_arr)

    # ---- Per-class classification report ---------------------------------
    report = classification_report(
        labels_arr, preds_arr, target_names=class_names, output_dict=True,
    )

    # ---- Class distribution ----------------------------------------------
    class_counts = {name: int((labels_arr == i).sum()) for i, name in enumerate(class_names)}

    # ---- Assemble result -------------------------------------------------
    avg_inference_ms = float(np.mean(batch_times)) * 1000

    result = {
        # Meta
        "config_path": str(config_path),
        "checkpoint_path": str(ckpt_path),
        "split": split,
        "num_images": num_images,
        "class_names": class_names,
        "class_counts": class_counts,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Model config
        "model_arch": cfg["model"]["arch"],
        "image_size": cfg["image_size"],
        "dropout": cfg["model"].get("dropout", 0.0),
        "seed": cfg["train"].get("seed", "N/A"),
        "epochs_trained": cfg["train"]["epochs"],
        "batch_size": cfg["train"]["batch_size"],
        "lr": cfg["train"]["lr"],
        "weight_decay": cfg["train"]["weight_decay"],
        "label_smoothing": cfg["train"].get("label_smoothing", 0.0),
        "decision_threshold": threshold,
        # Scalar metrics
        "accuracy": round(float(accuracy_score(labels_arr, preds_arr)), 4),
        "precision": round(float(precision_score(labels_arr, preds_arr, zero_division=0)), 4),
        "recall": round(sensitivity, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(float(f1_score(labels_arr, preds_arr, zero_division=0)), 4),
        "roc_auc": round(roc_auc, 4),
        # Confusion matrix [TN, FP, FN, TP]
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        # Per-class report
        "classification_report": {
            k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in v.items()}
            if isinstance(v, dict) else v
            for k, v in report.items()
        },
        # Curves
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "threshold_curve": threshold_curve,
        # Inference time
        "avg_inference_ms": round(avg_inference_ms, 2),
        # Raw predictions (for further analysis)
        "predictions": {
            "labels": labels_arr.tolist(),
            "probs": [round(p, 5) for p in probs_arr.tolist()],
        },
    }

    # ---- Persist to disk -------------------------------------------------
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(result, f, indent=2)

    return result


def load_cached_evaluation() -> dict | None:
    """Return the cached evaluation dict, or None if it doesn't exist."""
    if not CACHE_PATH.exists():
        return None
    with open(CACHE_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full evaluation and cache results.")
    parser.add_argument("--config", default="configs/cbis_ddsm_v3.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Running evaluation on '{args.split}' split ...")
    result = run_full_evaluation(args.config, args.split, args.threshold)
    print(f"  Accuracy:    {result['accuracy']:.4f}")
    print(f"  Sensitivity: {result['sensitivity']:.4f}")
    print(f"  Specificity: {result['specificity']:.4f}")
    print(f"  Precision:   {result['precision']:.4f}")
    print(f"  F1-score:    {result['f1']:.4f}")
    print(f"  ROC-AUC:     {result['roc_auc']:.4f}")
    print(f"  Avg infer:   {result['avg_inference_ms']:.1f} ms / image")
    cm = result["confusion_matrix"]
    print(f"  Confusion:   TN={cm['tn']}  FP={cm['fp']}  FN={cm['fn']}  TP={cm['tp']}")
    print(f"Cached to {CACHE_PATH}")

