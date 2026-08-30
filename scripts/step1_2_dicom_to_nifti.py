# -*- coding: utf-8 -*-

import os
import re
import sys
import json
import pandas as pd
from pathlib import Path
import pydicom
import SimpleITK as sitk


# ============================================================
# USER SETTINGS
# ============================================================

CSV_PATH = os.environ.get("DCE_PHASE_TABLE", "")
OUT_ROOT = os.environ.get("DCE_NIFTI_OUTPUT_ROOT", "")

VALID_SLICE_SUFFIXES = {".dcm"}

SORT_SLICES_BY_DICOM_Z = True
REQUIRE_SINGLE_SERIES_UID = True


# ============================================================
# SPECIAL STACKED MULTIPHASE CASES
# ============================================================

STACKED_CONFIG_PATH = os.environ.get("DCE_STACKED_CONFIG", "")
STACKED_SERIES_CASES = {}

if STACKED_CONFIG_PATH:
    with open(STACKED_CONFIG_PATH, "r", encoding="utf-8") as handle:
        STACKED_SERIES_CASES = json.load(handle)


# ============================================================
# HELPERS
# ============================================================

def resolve_to_dicom_dir(path_str):

    p = Path(str(path_str).strip())

    if not p.exists():

        raise FileNotFoundError(
            f"Path not found: {p}"
        )

    return p.parent if p.is_file() else p


def list_slice_files(data_dir):

    files = sorted(
        os.listdir(data_dir)
    )

    return [

        fn

        for fn in files

        if (data_dir / fn).is_file()

        and (data_dir / fn).suffix.lower()
        in VALID_SLICE_SUFFIXES
    ]


def list_slice_files_sorted_by_dicom(data_dir):

    slice_files = list_slice_files(
        data_dir
    )

    if len(slice_files) == 0:

        raise ValueError(
            f"No .dcm slice files found in {data_dir}"
        )

    meta = []

    series_uids = set()

    for fn in slice_files:

        fp = data_dir / fn

        ds = pydicom.dcmread(
            str(fp),
            stop_before_pixels=True,
            force=False
        )

        if (
            not hasattr(ds, "ImagePositionPatient")
            or ds.ImagePositionPatient is None
        ):

            raise ValueError(
                f"Missing ImagePositionPatient: {fp}"
            )

        z = float(
            ds.ImagePositionPatient[2]
        )

        instance = int(
            getattr(
                ds,
                "InstanceNumber",
                -1
            )
        )

        series_uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                ""
            )
        ).strip()

        if series_uid:

            series_uids.add(
                series_uid
            )

        meta.append(
            (
                z,
                instance,
                fn
            )
        )

    if (
        REQUIRE_SINGLE_SERIES_UID
        and len(series_uids) > 1
    ):

        raise ValueError(
            f"Multiple SeriesInstanceUID found "
            f"in {data_dir}: "
            f"{len(series_uids)}"
        )

    meta.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2]
        )
    )

    return [
        x[2]
        for x in meta
    ]


def natural_key(s):

    return [

        int(t)
        if t.isdigit()
        else t.lower()

        for t in re.split(
            r"(\d+)",
            s
        )
    ]


def list_slice_files_natural(
    data_dir
):

    files = []

    for fn in os.listdir(
        data_dir
    ):

        fp = data_dir / fn

        if (
            fp.is_file()
            and fp.suffix.lower()
            in VALID_SLICE_SUFFIXES
        ):

            files.append(
                fn
            )

    files.sort(
        key=natural_key
    )

    return files


def dicom_dir_to_raw_nifti(
    dicom_dir,
    output_path,
    files
):

    reader = (
        sitk.ImageSeriesReader()
    )

    reader.SetFileNames(
        [
            str(
                dicom_dir / fn
            )

            for fn in files
        ]
    )

    image = reader.Execute()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sitk.WriteImage(
        image,
        str(output_path)
    )


def extract_z_chunk(
    img3d,
    z_start,
    z_len
):

    size = list(
        img3d.GetSize()
    )

    if (
        z_start < 0
        or z_start + z_len > size[2]
    ):

        raise ValueError(
            f"Chunk outside image: "
            f"start={z_start}, "
            f"length={z_len}, "
            f"total_z={size[2]}"
        )

    extract_size = [
        size[0],
        size[1],
        z_len
    ]

    extract_index = [
        0,
        0,
        z_start
    ]

    return sitk.Extract(
        img3d,
        extract_size,
        extract_index
    )


# ============================================================
# NORMAL NON-STACKED PHASE EXPORT
# ============================================================

def export_normal_phase(
    session,
    phase_name,
    dicom_value,
    output_path
):

    # Missing path
    if (
        pd.isna(dicom_value)
        or str(dicom_value).strip() == ""
    ):

        print(
            f"[SKIP] {session}: "
            f"{phase_name} path missing"
        )

        return

    # Existing output
    if output_path.exists():

        print(
            f"[SKIP] {phase_name} already exists:"
        )

        print(
            output_path
        )

        return

    dicom_dir = (
        resolve_to_dicom_dir(
            str(dicom_value)
        )
    )

    print(
        f"{phase_name} DICOM: "
        f"{dicom_dir}"
    )

    if SORT_SLICES_BY_DICOM_Z:

        files = (
            list_slice_files_sorted_by_dicom(
                dicom_dir
            )
        )

    else:

        files = list_slice_files(
            dicom_dir
        )

    print(
        f"{phase_name}: "
        f"found {len(files)} slices"
    )

    dicom_dir_to_raw_nifti(
        dicom_dir,
        output_path,
        files
    )

    print(
        f"[OK] wrote {phase_name}:"
    )

    print(
        output_path
    )


# ============================================================
# STACKED MULTIPHASE CASE
# ============================================================

def process_stacked_subject(
    session,
    config,
    subject_out
):

    stacked_dir = Path(
        config["stacked_dir"]
    )

    chunk_size = int(
        config["chunk_size"]
    )

    num_phases = int(
        config["num_phases"]
    )

    reverse_files = bool(
        config.get(
            "reverse_files",
            False
        )
    )

    print(
        f"[STACKED] directory: "
        f"{stacked_dir}"
    )

    # --------------------------------------------------------
    # Check outputs
    # --------------------------------------------------------

    pre_output = (
        subject_out
        / "pre.nii.gz"
    )

    po1_output = (
        subject_out
        / "po1.nii.gz"
    )

    po2_output = (
        subject_out
        / "po2.nii.gz"
    )

    # If all three already exist,
    # don't even read the stacked DICOM.
    if (
        pre_output.exists()
        and po1_output.exists()
        and po2_output.exists()
    ):

        print(
            "[SKIP] PRE, PO1 and PO2 "
            "already exist."
        )

        return

    # --------------------------------------------------------
    # Read stacked DICOM series
    # --------------------------------------------------------

    all_files = (
        list_slice_files_natural(
            stacked_dir
        )
    )

    if reverse_files:

        all_files = list(
            reversed(
                all_files
            )
        )

    if len(all_files) == 0:

        raise ValueError(
            f"No DICOM slices found "
            f"in {stacked_dir}"
        )

    reader = (
        sitk.ImageSeriesReader()
    )

    reader.SetFileNames(
        [
            str(
                stacked_dir / fn
            )

            for fn in all_files
        ]
    )

    img_all = reader.Execute()

    size_xyz = (
        img_all.GetSize()
    )

    expected_total = (
        chunk_size
        * num_phases
    )

    if size_xyz[2] != expected_total:

        raise ValueError(
            f"STACKED case: expected "
            f"{expected_total} slices "
            f"({num_phases} x {chunk_size}) "
            f"but found {size_xyz[2]}"
        )

    subject_out.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # PRE = phase 0
    # --------------------------------------------------------

    if pre_output.exists():

        print(
            f"[SKIP] PRE already exists:"
        )

        print(
            pre_output
        )

    else:

        img_pre = extract_z_chunk(
            img_all,
            z_start=0,
            z_len=chunk_size
        )

        sitk.WriteImage(
            img_pre,
            str(pre_output)
        )

        print(
            f"[STACKED OK] wrote PRE:"
        )

        print(
            pre_output
        )

    # --------------------------------------------------------
    # PO1 = phase 1
    # --------------------------------------------------------

    if po1_output.exists():

        print(
            f"[SKIP] PO1 already exists:"
        )

        print(
            po1_output
        )

    else:

        img_po1 = extract_z_chunk(
            img_all,
            z_start=1 * chunk_size,
            z_len=chunk_size
        )

        sitk.WriteImage(
            img_po1,
            str(po1_output)
        )

        print(
            f"[STACKED OK] wrote PO1:"
        )

        print(
            po1_output
        )

    # --------------------------------------------------------
    # PO2 = phase 2
    # --------------------------------------------------------

    if po2_output.exists():

        print(
            f"[SKIP] PO2 already exists:"
        )

        print(
            po2_output
        )

    else:

        img_po2 = extract_z_chunk(
            img_all,
            z_start=2 * chunk_size,
            z_len=chunk_size
        )

        sitk.WriteImage(
            img_po2,
            str(po2_output)
        )

        print(
            f"[STACKED OK] wrote PO2:"
        )

        print(
            po2_output
        )


# ============================================================
# MAIN
# ============================================================

def main():

    if not CSV_PATH or not OUT_ROOT:
        raise RuntimeError(
            "Set DCE_PHASE_TABLE and DCE_NIFTI_OUTPUT_ROOT before running."
        )

    # SLURM array task index
    row_index = int(
        sys.argv[1]
    )

    # --------------------------------------------------------
    # Read mapping spreadsheet
    # --------------------------------------------------------

    if CSV_PATH.lower().endswith(
        (".xlsx", ".xls")
    ):

        df = pd.read_excel(
            CSV_PATH,
            engine="openpyxl"
        )

    else:

        df = pd.read_csv(
            CSV_PATH
        )

    # Remove accidental spaces
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Array bounds
    # --------------------------------------------------------

    if row_index >= len(df):

        print(
            f"Index {row_index} "
            f"outside dataframe."
        )

        return

    row = df.iloc[
        row_index
    ]

    session = str(
        row["Session"]
    ).strip()

    print(
        "\n================================"
    )

    print(
        f"Processing row {row_index}: "
        f"{session}"
    )

    print(
        "================================"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    subject_out = (
        Path(OUT_ROOT)
        / "NIFTI_RAW"
        / session
    )

    # --------------------------------------------------------
    # Is this a stacked special case?
    # --------------------------------------------------------

    if session in STACKED_SERIES_CASES:

        process_stacked_subject(
            session=session,
            config=STACKED_SERIES_CASES[
                session
            ],
            subject_out=subject_out
        )

    else:

        # ----------------------------------------------------
        # Normal exceptional case:
        # PRE / PO1 / PO2 are separate DICOM series
        # ----------------------------------------------------

        pre_value = row[
            "Pre_Path"
        ]

        po1_value = row[
            "Post_1_Path"
        ]

        po2_value = row[
            "Post_2_Path"
        ]

        pre_output = (
            subject_out
            / "pre.nii.gz"
        )

        po1_output = (
            subject_out
            / "po1.nii.gz"
        )

        po2_output = (
            subject_out
            / "po2.nii.gz"
        )

        export_normal_phase(
            session=session,
            phase_name="PRE",
            dicom_value=pre_value,
            output_path=pre_output
        )

        export_normal_phase(
            session=session,
            phase_name="PO1",
            dicom_value=po1_value,
            output_path=po1_output
        )

        export_normal_phase(
            session=session,
            phase_name="PO2",
            dicom_value=po2_value,
            output_path=po2_output
        )

    print(
        f"\n[DONE] {session}"
    )


if __name__ == "__main__":
    main()
