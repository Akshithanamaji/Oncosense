from pathlib import Path

import torch
from torchvision import datasets, transforms


def build_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            # Mild zoom/shift instead of a hard Resize -- masses/calcs vary in
            # size and position, so this acts like scale augmentation.
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(data_dir: str, image_size: int, batch_size: int, num_workers: int, splits=("train", "val", "test")):
    """Expects data_dir/{train,val,test}/<class_name>/*.png (ImageFolder layout).

    Produced by prepare_idc.py, prepare_breakhis.py, or prepare_cbis_ddsm.py.
    Pass `splits` to only require a subset (e.g. training doesn't need "test"
    to exist yet if you haven't downloaded/prepared it).
    """
    data_dir = Path(data_dir)
    loaders = {}
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"Missing {split_dir}. Run the matching prepare_*.py script first."
            )
        is_train = split == "train"
        dataset = datasets.ImageFolder(split_dir, transform=build_transforms(image_size, is_train))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        loaders[split] = (dataset, loader)
    return loaders


def compute_class_weights(dataset):
    """Inverse-frequency class weights for imbalanced datasets (IDC, BreakHis)."""
    counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / (len(counts) * counts.clamp(min=1))
    return weights
