"""Train a classifier on IDC, BreakHis, CBIS-DDSM (or any ImageFolder-style dataset).

Usage:
    python src/train.py --config configs/cbis_ddsm.yaml
    python src/train.py --config configs/idc.yaml

If training gets interrupted (e.g. the machine sleeps mid-run), resume from
where it left off instead of starting over:
    python src/train.py --config configs/cbis_ddsm.yaml --resume
"""
import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

from datasets import compute_class_weights, get_dataloaders
from model import get_model, get_param_groups, set_backbone_requires_grad
from utils import AverageMeter, get_device, save_checkpoint, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_meter = AverageMeter()
    all_labels, all_probs = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            if scaler is not None:
                with torch.autocast(device_type=device.type):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            loss_meter.update(loss.item(), images.size(0))
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().tolist())

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")  # only one class present in a tiny batch/split

    return loss_meter.avg, acc, f1, auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume from the last saved epoch for this run_name")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"Using device: {device}")

    splits = get_dataloaders(
        cfg["data_dir"], cfg["image_size"], cfg["train"]["batch_size"], cfg["train"]["num_workers"],
        splits=("train", "val"),
    )
    train_ds, train_loader = splits["train"]
    val_ds, val_loader = splits["val"]
    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    arch = cfg["model"]["arch"]
    dropout = cfg["model"].get("dropout", 0.0)
    model = get_model(arch, cfg["num_classes"], cfg["model"]["pretrained"], dropout=dropout).to(device)

    label_smoothing = cfg["train"].get("label_smoothing", 0.0)
    if cfg["train"]["use_class_weights"]:
        weights = compute_class_weights(train_ds).to(device)
        print(f"Class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Warm up the randomly-initialized classifier head before touching the
    # pretrained backbone -- freeze_backbone_epochs=0 disables this and just
    # fine-tunes everything from epoch 1 (previous behavior).
    freeze_epochs = cfg["train"].get("freeze_backbone_epochs", 0)
    if freeze_epochs > 0 and arch != "simple_cnn":
        set_backbone_requires_grad(model, arch, False)
        print(f"Backbone frozen for the first {freeze_epochs} epoch(s); training classifier head only.")

    if arch != "simple_cnn":
        param_groups = get_param_groups(model, arch, cfg["train"]["lr"], backbone_lr_mult=0.1)
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    use_amp = cfg["train"].get("amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type) if use_amp else None

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = cfg["output"]["run_name"]
    best_ckpt_path = ckpt_dir / f"{run_name}_best.pth"
    last_ckpt_path = ckpt_dir / f"{run_name}_last.pth"
    log_path = log_dir / f"{run_name}.csv"

    best_val_auc = -1.0
    patience = cfg["train"]["early_stopping_patience"]
    epochs_without_improvement = 0
    start_epoch = 1

    if args.resume and last_ckpt_path.exists():
        state = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        start_epoch = state["epoch"] + 1
        best_val_auc = state["best_val_auc"]
        epochs_without_improvement = state["epochs_without_improvement"]
        print(f"Resuming from epoch {start_epoch} (best_val_auc so far: {best_val_auc:.4f})")
    elif args.resume:
        print(f"--resume given but no checkpoint found at {last_ckpt_path}; starting fresh.")

    if freeze_epochs > 0 and arch != "simple_cnn" and start_epoch > freeze_epochs:
        # Resuming past the warmup point -- backbone should already be unfrozen.
        set_backbone_requires_grad(model, arch, True)

    file_mode = "a" if (args.resume and log_path.exists() and start_epoch > 1) else "w"
    with open(log_path, file_mode, newline="") as f:
        writer = csv.writer(f)
        if file_mode == "w":
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_f1", "val_auc"])

        for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
            if freeze_epochs > 0 and epoch == freeze_epochs + 1 and arch != "simple_cnn":
                set_backbone_requires_grad(model, arch, True)
                print(f"Epoch {epoch}: unfreezing backbone (lr_mult=0.1 vs head lr).")

            train_loss, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            val_loss, val_acc, val_f1, val_auc = run_epoch(model, val_loader, criterion, device)

            print(
                f"Epoch {epoch}/{cfg['train']['epochs']} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_auc={val_auc:.4f}"
            )
            writer.writerow([epoch, train_loss, val_loss, val_acc, val_f1, val_auc])
            f.flush()

            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                epochs_without_improvement = 0
                save_checkpoint(model, best_ckpt_path)
                print(f"  -> new best (val_auc={val_auc:.4f}), saved to {best_ckpt_path}")
            else:
                epochs_without_improvement += 1

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_auc": best_val_auc,
                "epochs_without_improvement": epochs_without_improvement,
            }, last_ckpt_path)

            if epochs_without_improvement >= patience:
                print(f"Early stopping: no val_auc improvement in {patience} epochs.")
                break

    print(f"Best val AUC: {best_val_auc:.4f}. Checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
