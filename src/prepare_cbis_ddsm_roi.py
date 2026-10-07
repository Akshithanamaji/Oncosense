"""Build an ImageFolder-style benign/malignant dataset from CBIS-DDSM
ROI-CROPPED LESION PATCHES, instead of full mammograms (see prepare_cbis_ddsm.py
for the full-image version).

Why: classifying a whole downsampled mammogram means the model has to both
*find* a small mass/calcification and classify it -- CBIS-DDSM full-mammogram
baselines plateau around 65-75% accuracy for this reason. Classifying the
already-cropped lesion patch is a much easier task because the model only has
to look at tissue that's already known to be the abnormality.

Requires an ADDITIONAL TCIA download beyond the full-mammogram manifests, OR
(much faster) src/fast_download_roi_series.py against the same --dicom-root.

CBIS-DDSM data quality quirk this script works around: each abnormality has
two IDC series -- a small cropped lesion patch and a full-mammogram-sized
binary ROI mask -- but which SeriesDescription ("cropped images" vs "ROI mask
images") is actually which is NOT reliable. Some "cropped images" series are
actually the full-size mask, and some "ROI mask images" series are actually
the small real patch. Trusting the SeriesDescription label alone silently
poisons ~half of training with near-blank resized masks (confirmed by
inspection: pixel arrays thousands of px across with ~2 distinct values).

So for each abnormality we look at BOTH series (whichever exist) and pick
whichever candidate .dcm is actually small and textured (real patch). If
neither candidate qualifies -- i.e. only the full-res mask is available,
which is the common case, since the genuine small "cropped images" series
is only hosted for about 60% of abnormalities -- we fall back to using the
mask as a bounding box: threshold it, find the lesion's bounding box, and
crop that region directly out of the corresponding FULL mammogram DICOM
(already downloaded for the whole-image classifier). This recovers a real,
correctly-cropped lesion patch for abnormalities that have no direct small
series at all.

Usage:
    python src/prepare_cbis_ddsm_roi.py ^
        --dicom-root "D:\\TCIA_Data_Retriever_windows_amd64\\cbis_ddsm" ^
        --train-csvs "...\\mass_case_description_train_set.csv" "...\\calc_case_description_train_set.csv" ^
        --test-csvs  "...\\mass_case_description_test_set.csv" "...\\calc_case_description_test_set.csv" ^
        --out-dir data/processed/cbis_ddsm_roi
"""
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split

from prepare_cbis_ddsm import MALIGNANT_VALUES, BENIGN_VALUES, crop_to_breast_region

MAX_MASK_UNIQUE_VALUES = 10  # binary/near-binary masks have very few distinct pixel values
MAX_PATCH_DIM = 1500  # a real lesion patch is at most a few hundred px; full mammograms are thousands
MASK_BBOX_MARGIN_FRAC = 0.15  # extra context around the mask's bounding box when cropping the full image


def load_series_lookups():
    """PatientID -> crdc_series_uuid, separately for the "cropped images" and
    "ROI mask images" SeriesDescriptions, straight from the IDC index."""
    from idc_index import index

    client = index.IDCClient()
    df = client.index
    cbis = df[df["collection_id"] == "cbis_ddsm"]
    crop = cbis[cbis["SeriesDescription"] == "cropped images"]
    mask = cbis[cbis["SeriesDescription"] == "ROI mask images"]
    return (
        dict(zip(crop["PatientID"], crop["crdc_series_uuid"])),
        dict(zip(mask["PatientID"], mask["crdc_series_uuid"])),
    )


def image_key_from_path(raw_path: str) -> str:
    return re.split(r"[\\/]", raw_path.strip())[0]


def load_abnormality_rows(csv_paths):
    """Returns list of (key, full_image_key, label, patient_id). Each row is
    its own abnormality/crop, so unlike the full-image script no
    malignant-wins merging is needed."""
    out = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for _, r in df.iterrows():
            pathology = str(r["pathology"]).strip()
            if pathology in MALIGNANT_VALUES:
                label = "malignant"
            elif pathology in BENIGN_VALUES:
                label = "benign"
            else:
                continue
            key = image_key_from_path(r["cropped image file path"])
            full_key = image_key_from_path(r["image file path"])
            out.append((key, full_key, label, r["patient_id"]))
    return out


def series_dicoms(dicom_root: Path, key: str, series_uuid):
    if series_uuid is None:
        return []
    folder = dicom_root / key / series_uuid
    if not folder.exists():
        return []
    return list(folder.rglob("*.dcm"))


def find_real_patch(dicom_root: Path, key: str, crop_lookup: dict, mask_lookup: dict):
    """Look at both of an abnormality's series (whichever exist) and return
    the .dcm that's actually a small, textured lesion patch -- regardless of
    which SeriesDescription it was filed under (see module docstring)."""
    candidates = (
        series_dicoms(dicom_root, key, crop_lookup.get(key))
        + series_dicoms(dicom_root, key, mask_lookup.get(key))
    )
    best = None  # (n_unique, path)
    for p in candidates:
        try:
            arr = pydicom.dcmread(p).pixel_array
        except Exception:
            continue
        if max(arr.shape[:2]) >= MAX_PATCH_DIM:
            continue  # full-mammogram-sized; not a patch
        n_unique = len(np.unique(arr))
        if n_unique <= MAX_MASK_UNIQUE_VALUES:
            continue  # binary/near-binary; a mask, not a real patch
        if best is None or n_unique > best[0]:
            best = (n_unique, p)
    return best[1] if best else None


def crop_from_full_via_mask(dicom_root: Path, key: str, full_key: str, mask_lookup: dict):
    """Fallback when no direct small patch is available: threshold the
    full-res ROI mask, find the lesion's bounding box, and crop that region
    out of the corresponding full mammogram. Returns a loaded pydicom
    Dataset and the cropped pixel array, or None if anything doesn't line up
    (mask/full-image dimension mismatch, nothing downloaded, empty mask)."""
    mask_dcms = series_dicoms(dicom_root, key, mask_lookup.get(key))
    full_folder = dicom_root / full_key
    full_dcms = list(full_folder.rglob("*.dcm")) if full_folder.exists() else []
    if not mask_dcms or not full_dcms:
        return None

    try:
        mask_arr = pydicom.dcmread(mask_dcms[0]).pixel_array
        full_ds = pydicom.dcmread(full_dcms[0])
        full_arr = full_ds.pixel_array
    except Exception:
        return None

    if mask_arr.shape != full_arr.shape:
        return None  # can't align a mask to a full image of a different size

    ys, xs = np.where(mask_arr > 0)
    if len(ys) == 0:
        return None

    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    my = int((y1 - y0) * MASK_BBOX_MARGIN_FRAC) + 10
    mx = int((x1 - x0) * MASK_BBOX_MARGIN_FRAC) + 10
    y0, x0 = max(0, y0 - my), max(0, x0 - mx)
    y1, x1 = min(full_arr.shape[0] - 1, y1 + my), min(full_arr.shape[1] - 1, x1 + mx)

    return full_ds, full_arr[y0:y1 + 1, x0:x1 + 1]


def array_to_pil(ds, arr: np.ndarray, size: int) -> Image.Image:
    arr = arr.astype(np.float32)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val * 255.0
    arr = arr.astype(np.uint8)
    arr = crop_to_breast_region(arr, margin=5)

    img = Image.fromarray(arr, mode="L").convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


def _convert_one(args):
    key, full_key, dicom_root, out_path, size, crop_lookup, mask_lookup = args

    patch_path = find_real_patch(dicom_root, key, crop_lookup, mask_lookup)
    if patch_path is not None:
        ds = pydicom.dcmread(patch_path)
        img = array_to_pil(ds, ds.pixel_array, size)
    else:
        fallback = crop_from_full_via_mask(dicom_root, key, full_key, mask_lookup)
        if fallback is None:
            return key, "missing"
        ds, arr = fallback
        img = array_to_pil(ds, arr, size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return key, "ok"


def process_rows(rows, patients_subset, dicom_root, out_dir, split_name, size, crop_lookup, mask_lookup, workers=8):
    jobs = []
    n_skipped = 0
    for key, full_key, label, patient_id in rows:
        if patient_id not in patients_subset:
            continue
        out_path = out_dir / split_name / label / f"{key}.png"
        if out_path.exists():
            n_skipped += 1
            continue
        jobs.append((key, full_key, dicom_root, out_path, size, crop_lookup, mask_lookup))

    if n_skipped:
        print(f"  ({n_skipped} already converted, skipped)")

    n_written, n_missing = 0, 0
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_convert_one, job) for job in jobs]
            for fut in as_completed(futures):
                _, status = fut.result()
                if status == "ok":
                    n_written += 1
                else:
                    n_missing += 1

    return n_written, n_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", required=True,
                         help="Folder of downloaded <key>/<series_uuid>/*.dcm -- same root as the "
                              "full-mammogram download works fine, folder names don't collide")
    parser.add_argument("--train-csvs", nargs="+", required=True)
    parser.add_argument("--test-csvs", nargs="*", default=[])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--size", type=int, default=224, help="Cropped patches are already tight, 224 is plenty")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dicom_root = Path(args.dicom_root)
    out_dir = Path(args.out_dir)

    print("Querying IDC index for cropped-image / ROI-mask series per abnormality...")
    crop_lookup, mask_lookup = load_series_lookups()
    print(f"  {len(crop_lookup)} abnormalities have a 'cropped images' series, "
          f"{len(mask_lookup)} have a 'ROI mask images' series.")

    train_rows = load_abnormality_rows(args.train_csvs)
    test_rows = load_abnormality_rows(args.test_csvs) if args.test_csvs else []

    train_patients = sorted({p for _, _, _, p in train_rows})
    if len(train_patients) < 2:
        raise SystemExit("Not enough patients in --train-csvs to split into train/val.")
    train_p, val_p = train_test_split(train_patients, test_size=args.val_frac, random_state=args.seed)
    train_p, val_p = set(train_p), set(val_p)

    n_tr, miss_tr = process_rows(train_rows, train_p, dicom_root, out_dir, "train", args.size, crop_lookup, mask_lookup, args.workers)
    n_val, miss_val = process_rows(train_rows, val_p, dicom_root, out_dir, "val", args.size, crop_lookup, mask_lookup, args.workers)

    print(f"train: {n_tr} written ({miss_tr} not downloaded/unreadable)")
    print(f"val:   {n_val} written ({miss_val} not downloaded/unreadable)")

    if test_rows:
        test_patients = {p for _, _, _, p in test_rows}
        n_test, miss_test = process_rows(test_rows, test_patients, dicom_root, out_dir, "test", args.size, crop_lookup, mask_lookup, args.workers)
        print(f"test:  {n_test} written ({miss_test} not downloaded/unreadable)")
    else:
        print("test:  skipped (no --test-csvs given).")


if __name__ == "__main__":
    main()
