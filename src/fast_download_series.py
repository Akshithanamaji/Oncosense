"""Fast parallel downloader for CBIS-DDSM series from the IDC open-data S3
bucket, using plain HTTPS GET instead of s5cmd. Direct HTTPS measured ~7MB/s
per connection here vs ~100-450KB/s effective through s5cmd's batch-run
mode -- so this fans out N concurrent HTTPS downloads with a thread pool
instead.

Usage:
    python src/fast_download_series.py --tag Mass-Test --dicom-root "D:\\...\\cbis_ddsm"
    python src/fast_download_series.py --tag Calc-Training,Calc-Test --dicom-root "D:\\...\\cbis_ddsm"
"""
import argparse
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from idc_index import index

BUCKET_URL = "https://idc-open-data.s3.amazonaws.com"


def list_key(series_uuid: str) -> str:
    resp = requests.get(BUCKET_URL, params={"list-type": "2", "prefix": f"{series_uuid}/"}, timeout=30)
    resp.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(resp.text)
    key_el = root.find(".//s3:Contents/s3:Key", ns)
    if key_el is None:
        raise RuntimeError(f"No object found for prefix {series_uuid}/")
    return key_el.text


def download_one(patient_id: str, series_uuid: str, dest_root: Path) -> str:
    out_dir = dest_root / patient_id / series_uuid
    if any(out_dir.glob("*.dcm")) if out_dir.exists() else False:
        return f"skip {patient_id}"
    key = list_key(series_uuid)
    filename = key.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / (filename + ".partial")
    with requests.get(f"{BUCKET_URL}/{key}", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    os.replace(tmp_path, out_dir / filename)  # atomic overwrite even if the destination already exists
    return f"ok {patient_id}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--tag", required=True, help="Comma-separated PatientID substrings, e.g. Mass-Test or Calc-Training,Calc-Test")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    dest_root = Path(args.dicom_root)
    tags = args.tag.split(",")
    pattern = "|".join(re.escape(t) for t in tags)

    client = index.IDCClient()
    df = client.index
    cbis = df[df["collection_id"] == "cbis_ddsm"]
    subset = cbis[
        cbis["PatientID"].str.contains(pattern, na=False, regex=True)
        & (cbis["SeriesDescription"] == "full mammogram images")
    ]

    todo = []
    for _, row in subset.iterrows():
        out_dir = dest_root / row["PatientID"] / row["crdc_series_uuid"]
        if out_dir.exists() and any(out_dir.glob("*.dcm")):
            continue
        todo.append((row["PatientID"], row["crdc_series_uuid"]))

    print(f"{len(subset)} total series for tag(s) {args.tag}; {len(todo)} remaining to download")
    if not todo:
        print("ALL_DOWNLOADED")
        return

    done, failed = 0, 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, pid, uid, dest_root): pid for pid, uid in todo}
        for fut in as_completed(futures):
            pid = futures[fut]
            try:
                fut.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"FAILED {pid}: {e}", file=sys.stderr)
            if (done + failed) % 20 == 0 or (done + failed) == len(todo):
                elapsed = time.time() - start
                print(f"progress: {done + failed}/{len(todo)} ({done} ok, {failed} failed) "
                      f"in {elapsed:.0f}s")

    print(f"ALL_DOWNLOADED (failed: {failed})")


if __name__ == "__main__":
    main()
