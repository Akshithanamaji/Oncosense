"""Split the IDC (Breast Histopathology Images) Kaggle dataset patient-wise.

Download from Kaggle first:
    kaggle datasets download -d paultimothymooney/breast-histopathology-images
Unzip so you have a folder of patient subfolders, each containing "0" and "1"
subfolders of 50x50 patches, e.g.:
    data/raw/idc/<patient_id>/0/*.png
    data/raw/idc/<patient_id>/1/*.png

We split by PATIENT (not by image) so patches from the same patient never
appear in both train and test -- splitting by image leaks patient-specific
texture and inflates accuracy.

Usage:
    python src/prepare_idc.py --raw-dir data/raw/idc --out-dir data/processed/idc
"""
import argparse
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # hardlink: instant, no extra disk space (same volume)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Folder of <patient_id>/{0,1}/*.png")
    parser.add_argument("--out-dir", required=True, help="Output ImageFolder-style dataset")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    patient_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if not patient_dirs:
        raise SystemExit(f"No patient folders found under {raw_dir}")

    train_patients, temp_patients = train_test_split(
        patient_dirs, test_size=args.val_frac + args.test_frac, random_state=args.seed
    )
    rel_test = args.test_frac / (args.val_frac + args.test_frac)
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=rel_test, random_state=args.seed
    )

    splits = {"train": train_patients, "val": val_patients, "test": test_patients}
    print(f"Patients -> train: {len(train_patients)}, val: {len(val_patients)}, test: {len(test_patients)}")

    for split_name, patients in splits.items():
        n_images = 0
        for patient_dir in patients:
            for label in ["0", "1"]:
                label_dir = patient_dir / label
                if not label_dir.exists():
                    continue
                for img_path in label_dir.glob("*.png"):
                    dst = out_dir / split_name / label / f"{patient_dir.name}_{img_path.name}"
                    link_or_copy(img_path, dst)
                    n_images += 1
        print(f"{split_name}: {n_images} images written to {out_dir / split_name}")


if __name__ == "__main__":
    main()
