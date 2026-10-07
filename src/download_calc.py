"""Download CBIS-DDSM Calc-Training and Calc-Test full mammogram images via
the NCI Imaging Data Commons (IDC) index client. Adds calcification cases
alongside the existing mass cases, to grow the training set and give a
combined mass+calc test set for final evaluation.

Usage:
    python src/download_calc.py --dicom-root "D:\\TCIA_Data_Retriever_windows_amd64\\cbis_ddsm"
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
    calc = cbis[
        cbis["PatientID"].str.contains("Calc-Training|Calc-Test", na=False, regex=True)
        & (cbis["SeriesDescription"] == "full mammogram images")
    ]
    series_uids = calc["SeriesInstanceUID"].tolist()
    print(f"{len(series_uids)} Calc-Training/Calc-Test full-mammogram series to download "
          f"({calc['series_size_MB'].sum() / 1024:.1f} GB)")

    client.download_dicom_series(
        seriesInstanceUID=series_uids,
        downloadDir=args.dicom_root,
        use_s5cmd_sync=True,
        dirTemplate="%PatientID/%SeriesInstanceUID",
        show_progress_bar=True,
    )
    print("DOWNLOAD_DONE")


if __name__ == "__main__":
    main()
