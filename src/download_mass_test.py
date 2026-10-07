"""Download CBIS-DDSM Mass-Test full mammogram images via the NCI Imaging
Data Commons (IDC) index client -- no TCIA Data Retriever GUI needed.

Downloads into the same folder layout the manual TCIA Retriever used
(<PatientID>/<SeriesInstanceUID>/*.dcm), so prepare_cbis_ddsm.py's
find_dicom() picks them up unchanged.

Usage:
    python src/download_mass_test.py --dicom-root "D:\\TCIA_Data_Retriever_windows_amd64\\cbis_ddsm"
"""
import argparse

from idc_index import index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom-root", required=True)
    args = parser.parse_args()

    client = index.IDCClient()
    df = client.index
    cbis = df[df["collection_id"] == "cbis_ddsm"]
    mass_test = cbis[
        cbis["PatientID"].str.contains("Mass-Test", na=False)
        & (cbis["SeriesDescription"] == "full mammogram images")
    ]
    series_uids = mass_test["SeriesInstanceUID"].tolist()
    print(f"{len(series_uids)} Mass-Test full-mammogram series to download "
          f"({mass_test['series_size_MB'].sum() / 1024:.1f} GB)")

    client.download_dicom_series(
        seriesInstanceUID=series_uids,
        downloadDir=args.dicom_root,
        use_s5cmd_sync=True,  # skips files already downloaded -- safe to re-run after an interruption
        dirTemplate="%PatientID/%SeriesInstanceUID",
        show_progress_bar=True,
    )
    print("DOWNLOAD_DONE")


if __name__ == "__main__":
    main()
