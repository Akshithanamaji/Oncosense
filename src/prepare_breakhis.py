"""Split the BreakHis dataset patient-wise into benign/malignant train/val/test folders.

Download from Kaggle first, e.g.:
    kaggle datasets download -d ambarish/breakhis
Standard BreakHis layout (v1) looks like:
    BreaKHis_v1/histology_slides/breast/benign/SOB/<tumor_type>/<patient_id>/<mag>/*.png
    BreaKHis_v1/histology_slides/breast/malignant/SOB/<tumor_type>/<patient_id>/<mag>/*.png
i.e. each image's parent folder is the magnification (40X/100X/200X/400X) and
the grandparent folder is the patient id.

We split by PATIENT so slides from the same patient never span train/test,
and we filter to a single magnification (mixing magnifications in one split
makes the task inconsistent).

Usage:
    python src/prepare_breakhis.py --raw-dir data/raw/breakhis --out-dir data/processed/breakhis --magnification 40X
"""
import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def classify(path: Path):
    parts_lower = [p.lower() for p in path.parts]
    if "benign" in parts_lower:
        return "benign"
    if "malignant" in parts_lower:
        return "malignant"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--magnification", default="40X", choices=["40X", "100X", "200X", "400X"])
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    # patient_id -> list of (image_path, class_label)
    patient_images = defaultdict(list)
    for img_path in raw_dir.rglob("*"):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if img_path.parent.name != args.magnification:
            continue
        label = classify(img_path)
        if label is None:
            continue
        patient_id = img_path.parent.parent.name
        patient_images[patient_id].append((img_path, label))

    if not patient_images:
        raise SystemExit(
            f"No images found for magnification={args.magnification} under {raw_dir}. "
            "Check the folder layout matches BreakHis_v1/.../<patient_id>/<mag>/*.png"
        )

    patients = sorted(patient_images.keys())
    train_p, temp_p = train_test_split(patients, test_size=args.val_frac + args.test_frac, random_state=args.seed)
    rel_test = args.test_frac / (args.val_frac + args.test_frac)
    val_p, test_p = train_test_split(temp_p, test_size=rel_test, random_state=args.seed)

    splits = {"train": train_p, "val": val_p, "test": test_p}
    print(f"Patients -> train: {len(train_p)}, val: {len(val_p)}, test: {len(test_p)}")

    for split_name, plist in splits.items():
        n_images = 0
        for patient_id in plist:
            for img_path, label in patient_images[patient_id]:
                dst = out_dir / split_name / label / f"{patient_id}_{img_path.name}"
                link_or_copy(img_path, dst)
                n_images += 1
        print(f"{split_name}: {n_images} images written to {out_dir / split_name}")


if __name__ == "__main__":
    main()
