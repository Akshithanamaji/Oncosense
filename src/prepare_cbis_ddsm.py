"""Build an ImageFolder-style benign/malignant dataset from CBIS-DDSM.

CBIS-DDSM ships as two pieces that have to be joined yourself:
  1. DICOM images, downloaded via the TCIA Data Retriever from a .tcia
     manifest (e.g. "Mass-Training_full_mammogram_images.tcia"). Each image
     lands in its own folder named like
         Mass-Training_P_00001_LEFT_CC/<StudyUID>/<SeriesUID>/<uuid>.dcm
  2. CSV case description files (mass/calc x train/test) from the CBIS-DDSM
     page, with a "pathology" column (MALIGNANT / BENIGN /
     BENIGN_WITHOUT_CALLBACK) and an "image file path" column whose first
     path segment is exactly that folder name.

This script joins the two: for every row in the CSVs you pass in, it finds
the matching downloaded DICOM, converts it to a windowed 8-bit PNG, and
writes it into data/processed/cbis_ddsm/{train,val,test}/{benign,malignant}/.

It's safe to run with a PARTIAL download -- rows whose DICOM hasn't been
downloaded yet are skipped and counted, not treated as an error. Re-run the
same command after downloading more .tcia manifests to pick up the rest.

Patient-level split: rows from --train-csvs are pooled and split
patient-wise into train/val. Rows from --test-csvs (if any are downloaded)
go entirely to test, matching the official CBIS-DDSM benchmark split -- so
your test numbers stay comparable to published results. If you don't have
any test-set images yet, omit --test-csvs and the test folder is simply
left empty until you do.

Usage (with what you have today -- Mass-Training full mammograms only):
    python src/prepare_cbis_ddsm.py ^
        --dicom-root "D:\\TCIA_Data_Retriever_windows_amd64\\cbis_ddsm" ^
        --train-csvs "C:\\Users\\HP\\Downloads\\mass_case_description_train_set.csv" ^
        --out-dir data/processed/cbis_ddsm

Once you've also downloaded Calc-Training, Mass-Test, Calc-Test full
mammogram images:
    python src/prepare_cbis_ddsm.py ^
        --dicom-root "D:\\TCIA_Data_Retriever_windows_amd64\\cbis_ddsm" ^
        --train-csvs "...\\mass_case_description_train_set.csv" "...\\calc_case_description_train_set.csv" ^
        --test-csvs  "...\\mass_case_description_test_set.csv" "...\\calc_case_description_test_set.csv" ^
        --out-dir data/processed/cbis_ddsm
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split

MALIGNANT_VALUES = {"MALIGNANT"}
BENIGN_VALUES = {"BENIGN", "BENIGN_WITHOUT_CALLBACK"}


def image_key_from_path(raw_path: str) -> str:
    return re.split(r"[\\/]", raw_path.strip())[0]


def load_case_rows(csv_paths):
    """Returns dict: image_key -> (label, patient_id). Multiple abnormalities
    per image are resolved malignant-wins (if any abnormality in the image
    is malignant, the whole image is labeled malignant)."""
    rows = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            key = image_key_from_path(r["image file path"])
            pathology = str(r["pathology"]).strip()
            if pathology in MALIGNANT_VALUES:
                label = "malignant"
            elif pathology in BENIGN_VALUES:
                label = "benign"
            else:
                continue  # unrecognized pathology value, skip

            patient_id = r["patient_id"]
            if key in rows and rows[key][0] == "malignant":
                continue  # already malignant, keep it (malignant wins)
            rows[key] = (label, patient_id)
    return rows


def crop_to_breast_region(arr: np.ndarray, margin: int = 20) -> np.ndarray:
    """Crop out the black background/border, keeping only the breast tissue.

    The background isn't literally zero -- mammograms have a diffuse
    low-level scatter/fog that's visually black but numerically nonzero, so
    a fixed low threshold connects the whole background (plus scanner
    markers and laterality/view text labels sitting in it) into one giant
    blob spanning nearly the whole image. Otsu's method picks a
    per-image adaptive cutoff that actually separates tissue from that fog,
    and morphological opening breaks thin residual connections (border
    frame lines) before we take the largest connected component. Without
    this, the model can learn to key off background/corner artifacts
    instead of the actual tissue (visible as border/corner hotspots in
    Grad-CAM)."""
    _, mask = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return arr  # nothing found, fall back to the full image

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[largest_label, :4]
    y0, x0 = max(0, y - margin), max(0, x - margin)
    y1, x1 = min(arr.shape[0], y + h + margin), min(arr.shape[1], x + w + margin)
    return arr[y0:y1, x0:x1]


def dicom_to_pil(ds_or_path, size: int) -> Image.Image:
    """Shared preprocessing: DICOM -> windowed, breast-cropped, resized PIL image.

    Accepts either a path to a .dcm file or an already-loaded pydicom Dataset
    (the latter lets callers like the Streamlit app read from an in-memory
    upload without touching disk).
    """
    ds = ds_or_path if hasattr(ds_or_path, "pixel_array") else pydicom.dcmread(ds_or_path)
    arr = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val * 255.0
    arr = arr.astype(np.uint8)
    arr = crop_to_breast_region(arr)

    # Resize (stretch) the cropped breast region straight to a square --
    # deliberately NOT padding to square first. Zero-padding leaves a black
    # margin whose thickness/position varies systematically with image
    # aspect ratio, and the model can (and does, confirmed via Grad-CAM)
    # learn to key off that margin as a shortcut instead of the tissue.
    # Stretching introduces mild aspect distortion but no artificial edge
    # for the model to exploit.
    img = Image.fromarray(arr, mode="L").convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


def dicom_to_png(dcm_path: Path, out_path: Path, size: int):
    img = dicom_to_pil(dcm_path, size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def find_dicom(dicom_root: Path, image_key: str):
    folder = dicom_root / image_key
    if not folder.exists():
        return None
    matches = list(folder.rglob("*.dcm"))
    return matches[0] if matches else None


def _convert_one(args):
    dcm_path, out_path, size = args
    dicom_to_png(dcm_path, out_path, size)


def process_rows(rows_by_key, patients_subset, dicom_root, out_dir, split_name, size, workers=8):
    n_missing, n_skipped = 0, 0
    jobs = []
    for key, (label, patient_id) in rows_by_key.items():
        if patient_id not in patients_subset:
            continue
        out_path = out_dir / split_name / label / f"{key}.png"
        if out_path.exists():
            n_skipped += 1  # already converted in a previous (interrupted) run
            continue
        dcm_path = find_dicom(dicom_root, key)
        if dcm_path is None:
            n_missing += 1
            continue
        jobs.append((dcm_path, out_path, size))

    if n_skipped:
        print(f"  ({n_skipped} already converted, skipped)")

    n_written = 0
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_convert_one, job) for job in jobs]
            for fut in as_completed(futures):
                fut.result()  # surface any conversion errors immediately
                n_written += 1

    return n_written, n_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", required=True, help="Folder of downloaded <image_key>/.../*.dcm")
    parser.add_argument("--train-csvs", nargs="+", required=True, help="mass/calc *_train_set.csv files")
    parser.add_argument("--test-csvs", nargs="*", default=[], help="mass/calc *_test_set.csv files (optional)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-frac", type=float, default=0.15, help="Fraction of TRAIN patients held out for val")
    parser.add_argument("--size", type=int, default=512, help="Output PNG size (square, after padding)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes for DICOM->PNG conversion")
    args = parser.parse_args()

    dicom_root = Path(args.dicom_root)
    out_dir = Path(args.out_dir)

    train_rows = load_case_rows(args.train_csvs)
    test_rows = load_case_rows(args.test_csvs) if args.test_csvs else {}

    train_patients = sorted({p for _, p in train_rows.values()})
    if len(train_patients) < 2:
        raise SystemExit("Not enough patients in --train-csvs to split into train/val.")
    train_p, val_p = train_test_split(train_patients, test_size=args.val_frac, random_state=args.seed)
    train_p, val_p = set(train_p), set(val_p)

    n_tr, miss_tr = process_rows(train_rows, train_p, dicom_root, out_dir, "train", args.size, args.workers)
    n_val, miss_val = process_rows(train_rows, val_p, dicom_root, out_dir, "val", args.size, args.workers)

    print(f"train: {n_tr} images written ({miss_tr} rows had no downloaded DICOM yet)")
    print(f"val:   {n_val} images written ({miss_val} rows had no downloaded DICOM yet)")

    if test_rows:
        test_patients = {p for _, p in test_rows.values()}
        n_test, miss_test = process_rows(test_rows, test_patients, dicom_root, out_dir, "test", args.size, args.workers)
        print(f"test:  {n_test} images written ({miss_test} rows had no downloaded DICOM yet)")
    else:
        print("test:  skipped (no --test-csvs given). Download the Mass-Test / Calc-Test "
              "full-mammogram-image manifests and re-run with --test-csvs to fill this in "
              "before reporting final numbers.")


if __name__ == "__main__":
    main()
