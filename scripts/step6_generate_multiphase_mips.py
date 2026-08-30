# -*- coding: utf-8 -*-
"""
Step 6: Generate aligned PRE, PO1, and PO2 2D maximum-intensity
projections (MIPs) for one subject.

This script reads the N4-corrected, 1-mm volumes created by Step 5. PRE and
PO2 were resampled onto the PO1 reference grid in Step 5, so the same manual
tumor z-slice range can be applied to all three phases.

For each phase, the script:
    1. extracts the full in-plane slices from z_start through z_end;
    2. calculates a MIP across that z range;
    3. clips/scales the MIP using its 1st and 99th percentiles;
    4. converts it to uint8 and resizes it to 224 x 224 pixels;
    5. saves a phase-labelled PNG.

One invocation processes one row from the manual-range table. The row index
can be supplied with --row-index or automatically through
SLURM_ARRAY_TASK_ID when submitted as a SLURM array job.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk
from docx import Document
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE = os.environ.get("DCE_PIPELINE_ROOT", "")
MANUAL_DOCX = os.environ.get("DCE_MANUAL_RANGES_DOCX", "")

# Step 5 full-volume outputs. All three phases use the same PO1 1-mm grid.
STEP5_DIR = os.environ.get("DCE_STEP5_OUTPUT", "")

PHASE_DIRS = {
    "pre": os.path.join(STEP5_DIR, "debug_pre_BCR_N4_1mm_nifti"),
    "po1": os.path.join(STEP5_DIR, "debug_po1_BCR_N4_1mm_nifti"),
    "po2": os.path.join(STEP5_DIR, "debug_po2_BCR_N4_1mm_nifti"),
}

OUTPUT_DIR = os.environ.get("DCE_STEP6_OUTPUT", "")


# ============================================================
# READ MANUAL RANGES
# ============================================================

def read_manual_ranges():
    doc = Document(MANUAL_DOCX)
    table = doc.tables[0]

    headers = [cell.text.strip() for cell in table.rows[0].cells]
    subject_col = headers.index("Subject ID")
    range_col = headers.index("My range")

    rows = []

    for table_row in table.rows[1:]:
        cells = [cell.text.strip() for cell in table_row.cells]

        subject = cells[subject_col]
        manual_range = cells[range_col]

        subject_match = re.search(r"(M\d{8}_\d{14})", subject)
        numbers = re.findall(r"\d+", manual_range)

        if subject_match is None or len(numbers) < 2:
            continue

        z1 = int(numbers[0])
        z2 = int(numbers[1])

        rows.append(
            {
                "subject_id": subject_match.group(1),
                "z_start": min(z1, z2),
                "z_end": max(z1, z2),
            }
        )

    return (
        pd.DataFrame(rows)
        .drop_duplicates("subject_id", keep="last")
        .reset_index(drop=True)
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def normalize_uint8(image):
    """Apply the same 1st-99th percentile normalization as the old script."""
    image = image.astype(np.float32)

    lo = np.percentile(image, 1)
    hi = np.percentile(image, 99)

    image = (image - lo) / (hi - lo + 1e-8)
    image = np.clip(image, 0, 1)

    return (image * 255).astype(np.uint8)


def same_geometry(image_a, image_b, tolerance=1e-5):
    """Check that two SimpleITK images occupy the same voxel grid."""
    return (
        image_a.GetSize() == image_b.GetSize()
        and np.allclose(image_a.GetSpacing(), image_b.GetSpacing(), atol=tolerance)
        and np.allclose(image_a.GetOrigin(), image_b.GetOrigin(), atol=tolerance)
        and np.allclose(image_a.GetDirection(), image_b.GetDirection(), atol=tolerance)
    )


def get_row_index(command_line_index):
    """Use --row-index when provided; otherwise use the SLURM array index."""
    if command_line_index is not None:
        return command_line_index

    slurm_index = os.environ.get("SLURM_ARRAY_TASK_ID")

    if slurm_index is None:
        raise ValueError(
            "Provide --row-index for a manual run or submit as a SLURM array job."
        )

    return int(slurm_index)


# ============================================================
# MAIN
# ============================================================

def main():
    required_paths = {
        "DCE_MANUAL_RANGES_DOCX": MANUAL_DOCX,
        "DCE_STEP5_OUTPUT": STEP5_DIR,
        "DCE_STEP6_OUTPUT": OUTPUT_DIR,
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))

    parser = argparse.ArgumentParser()
    parser.add_argument("--row-index", type=int, default=None)
    args = parser.parse_args()

    row_index = get_row_index(args.row_index)
    manual_df = read_manual_ranges()

    if row_index < 0 or row_index >= len(manual_df):
        raise IndexError(
            f"row-index {row_index} is out of range. "
            f"The manual table contains {len(manual_df)} valid subject ranges."
        )

    row = manual_df.iloc[row_index]
    subject_id = row["subject_id"]
    original_z_start = int(row["z_start"])
    original_z_end = int(row["z_end"])

    # Load all three Step 5 volumes before creating any output.
    images = {}
    arrays = {}
    input_paths = {}

    for phase, phase_dir in PHASE_DIRS.items():
        input_path = os.path.join(
            phase_dir,
            f"{subject_id}_{phase}_N4_1mm.nii.gz",
        )

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Missing {phase.upper()} file: {input_path}")

        image = sitk.ReadImage(input_path, sitk.sitkFloat32)
        images[phase] = image
        arrays[phase] = sitk.GetArrayFromImage(image).astype(np.float32)
        input_paths[phase] = input_path

    # The identical grid is what makes one z range valid for all phases.
    for phase in ["pre", "po2"]:
        if not same_geometry(images["po1"], images[phase]):
            raise ValueError(
                f"{phase.upper()} geometry does not match PO1 for {subject_id}. "
                "Do not apply the same z indices to these volumes."
            )

    depth = arrays["po1"].shape[0]
    z_start = max(0, original_z_start)
    z_end = min(depth - 1, original_z_end)

    if z_start > z_end:
        raise ValueError(
            f"Invalid range for {subject_id}: {original_z_start}-{original_z_end}; "
            f"volume depth is {depth}."
        )

    span = z_end - z_start + 1
    saved_paths = {}

    for phase in ["pre", "po1", "po2"]:
        # Full-slice 3D slab: no x-y tumor cropping.
        slab = arrays[phase][z_start:z_end + 1, :, :]

        if slab.size == 0:
            raise ValueError(f"Empty {phase.upper()} slab for {subject_id}")

        mip = np.max(slab, axis=0)
        mip_uint8 = normalize_uint8(mip)

        mip_224 = Image.fromarray(mip_uint8).resize(
            (224, 224),
            Image.BILINEAR,
        )

        phase_output_dir = os.path.join(OUTPUT_DIR, phase)
        os.makedirs(phase_output_dir, exist_ok=True)

        output_path = os.path.join(
            phase_output_dir,
            (
                f"{subject_id}_{phase}_z{z_start:03d}_to_z{z_end:03d}_"
                f"span{span:03d}_fullslice_MIP.png"
            ),
        )

        mip_224.save(output_path)
        saved_paths[phase] = output_path

    print(f"[OK] {subject_id}")
    print(f"Manual range: {original_z_start}-{original_z_end}")
    print(f"Applied range: {z_start}-{z_end}")
    print(f"Slices used: {span}")

    for phase in ["pre", "po1", "po2"]:
        print(f"{phase.upper()} input: {input_paths[phase]}")
        print(f"{phase.upper()} MIP:   {saved_paths[phase]}")


if __name__ == "__main__":
    main()
