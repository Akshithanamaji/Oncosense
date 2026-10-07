"""Fill gaps in the IDC mirror by fetching specific CBIS-DDSM series directly
from TCIA's own NBIA REST API (authoritative source; IDC's public S3 mirror
is missing a chunk of the Calc-Test full-mammogram-image series).

Usage:
    python src/fetch_missing_from_tcia.py --dicom-root "D:\\...\\cbis_ddsm" --tag Calc-Test
"""
import argparse
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

NBIA_SERIES_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries"
NBIA_IMAGE_URL = "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage"
FULL_MAMMO_PATTERN = re.compile(r"^{tag}_P_\d+_(LEFT|RIGHT)_(CC|MLO)$")


def download_one(patient_id: str, series_uid: str, dest_root: Path) -> str:
    out_dir = dest_root / patient_id / series_uid
    if out_dir.exists() and any(out_dir.glob("*.dcm")):
        return f"skip {patient_id}"
    resp = requests.get(NBIA_IMAGE_URL, params={"SeriesInstanceUID": series_uid}, timeout=120)
    resp.raise_for_status()
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for name in z.namelist():
            if name.lower().endswith(".dcm"):
                out_dir.joinpath(Path(name).name).write_bytes(z.read(name))
    return f"ok {patient_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--tag", required=True, help="e.g. Calc-Test")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    dest_root = Path(args.dicom_root)

    resp = requests.get(NBIA_SERIES_URL, params={"Collection": "CBIS-DDSM"}, timeout=120)
    resp.raise_for_status()
    all_series = resp.json()

    pattern = re.compile(FULL_MAMMO_PATTERN.pattern.format(tag=re.escape(args.tag)))
    subset = [s for s in all_series if pattern.match(s["PatientID"]) and s["SeriesDescription"] == "full mammogram images"]

    todo = []
    for s in subset:
        out_dir = dest_root / s["PatientID"] / s["SeriesInstanceUID"]
        if out_dir.exists() and any(out_dir.glob("*.dcm")):
            continue
        todo.append((s["PatientID"], s["SeriesInstanceUID"]))

    print(f"{len(subset)} total {args.tag} full-mammogram series on TCIA; {len(todo)} remaining to download")
    if not todo:
        print("ALL_DOWNLOADED")
        return

    done, failed = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, pid, uid, dest_root): pid for pid, uid in todo}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                fut.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"FAILED {pid}: {e}")
            if (done + failed) % 10 == 0 or (done + failed) == len(todo):
                print(f"progress: {done + failed}/{len(todo)} ({done} ok, {failed} failed)")

    print(f"ALL_DOWNLOADED (failed: {failed})")


if __name__ == "__main__":
    main()
