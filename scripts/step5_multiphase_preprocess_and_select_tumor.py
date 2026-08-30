#Excel
#Best manually verified tumor slice
#              │
#              ▼
#       Original PO1 MRI
#              │
#        N4 correction
#              │
#        resample 1 mm
#              │
#              ├────────────────────┐
#              │                    │
#              │              Probability map
#              │                    │
#              │             resample to PO1
#              │                    │
#              │             probability > 0
#              │                    │
#              │             binary tumor mask
#              │                    │
#              │        manually guided ±2 slices
#              │                    │
#              │        connected components
#              │                    │
#              │     remove tiny components
#              │                    │
#              │    meanProb × sqrt(area)
#              │                    │
#              │      choose best component
#              │                    │
#              │          bounding box
#              │                    │
#              └──────────────┬─────┘
#                             ▼
#                    Crop actual PO1 MRI
#                             │
#                    min-max normalization
#                             │
#                       resize 224×224
#                             │
#                             ▼
#                  FINAL TUMOR PO1 CROP
#                  
                  
                  


# -*- coding: utf-8 -*-
import os
import re
import math
import json
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE = os.environ.get("DCE_PIPELINE_ROOT", "")

# Existing metadata containing radiologist-selected PO1 slice
EXCEL_PATH = os.environ.get("DCE_METADATA_XLSX", "")

# Combined Steps 1+2 output:
# contains pre.nii.gz, po1.nii.gz and po2.nii.gz
NIFTI_ROOT = os.environ.get("DCE_NIFTI_ROOT", "")

# Combined Steps 3+4 output:
# contains breast_mask.nii.gz, prob_1st.nii.gz and seg_2nd.nii.gz
SEGMENTATION_ROOT = os.environ.get("DCE_SEGMENTATION_ROOT", "")

# New Step 5 output.
# Use a new directory so old N4/1-mm files are not reused.
OUTPUT_DIR = os.environ.get("DCE_STEP5_OUTPUT", "")






EXCEL_SHEET_NAME = "Data"
BEST_SLICE_COL = "Best Slice (MR PO1)"
PRIMARY_SUBJECT_COL = "MIRRIR_DCE-MRI"
SECONDARY_SUBJECT_COL = "MIRRIR_DCE-MRI2"

SPECIAL_USE_SECONDARY_ONLY = {
    value.strip()
    for value in os.environ.get("DCE_SECONDARY_ONLY_IDS", "").split(",")
    if value.strip()
}

PRE_FILENAME = "pre.nii.gz"
PO1_FILENAME = "po1.nii.gz"
PO2_FILENAME = "po2.nii.gz"
PROB_FILENAME = "prob_1st.nii.gz"

SAVE_NPY = True
BBOX_THICKNESS = 3
MIN_COMPONENT_PIXELS = 20
PROB_THRESHOLD = 0.0   # keep same as your current prob > 0 logic


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def normalize_subject_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return None
    return s


def parse_best_slice(value):
    if pd.isna(value):
        return None

    s = str(value).strip()
    if s == "":
        return None

    s_upper = s.upper()

    if s_upper in {"N/A", "NA", "XXXX"}:
        return None

    if re.fullmatch(r"\d+", s):
        return int(s)

    m = re.match(r"^\s*(\d+)", s)
    if m:
        return int(m.group(1))

    nums = re.findall(r"\d+", s)
    if len(nums) == 0:
        return None

    if len(nums) > 1 and "95" in nums:
        return 95

    return int(nums[0])


def find_phase_paths(subject_id):
    subject_dir = os.path.join(NIFTI_ROOT, subject_id)

    paths = {
        "pre": os.path.join(subject_dir, PRE_FILENAME),
        "po1": os.path.join(subject_dir, PO1_FILENAME),
        "po2": os.path.join(subject_dir, PO2_FILENAME),
    }

    missing = [
        phase
        for phase, path in paths.items()
        if not os.path.exists(path)
    ]

    return paths, missing


def find_subject_segmentation_folder(subject_id):
    subject_dir = os.path.join(
        SEGMENTATION_ROOT,
        subject_id
    )

    if os.path.isdir(subject_dir):
        return subject_dir, "combined_step3_step4"

    return None, None


def find_probability_path(subject_dir):
    prob_path = os.path.join(subject_dir, PROB_FILENAME)
    if not os.path.exists(prob_path):
        raise FileNotFoundError(f"Missing probability file: {prob_path}")
    return prob_path


def n4_bias_field_correction(sitk_image):
    sitk_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = corrector.Execute(sitk_image, mask_image)
    return corrected


def resample_image_to_1mm(image, is_label=False):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_spacing = (1.0, 1.0, 1.0)

    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    )

    out = resampler.Execute(image)
    return out, original_size, new_size, original_spacing, new_spacing


def resample_probability_to_reference(prob_img, reference_img):
    """
    Resample Duke probability map into the exact geometry of the BCR-preprocessed PO1 image.
    Probability map is continuous, so use linear interpolation.
    """
    prob_img = sitk.Cast(prob_img, sitk.sitkFloat32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkLinear)

    prob_resampled = resampler.Execute(prob_img)
    return prob_resampled


def resample_image_to_reference(image, reference_img):
    """Resample an MRI volume into the exact PO1 1-mm reference grid."""
    image = sitk.Cast(image, sitk.sitkFloat32)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_img)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


def assert_same_geometry(image, reference_img, image_name, tolerance=1e-5):
    """Confirm that an output occupies the exact PO1 reference grid."""
    if image.GetSize() != reference_img.GetSize():
        raise ValueError(
            f"{image_name} size mismatch: "
            f"{image.GetSize()} versus {reference_img.GetSize()}"
        )

    geometry_fields = [
        ("spacing", image.GetSpacing(), reference_img.GetSpacing()),
        ("origin", image.GetOrigin(), reference_img.GetOrigin()),
        ("direction", image.GetDirection(), reference_img.GetDirection()),
    ]

    for field_name, values, reference_values in geometry_fields:
        if any(
            abs(float(a) - float(b)) > tolerance
            for a, b in zip(values, reference_values)
        ):
            raise ValueError(
                f"{image_name} {field_name} mismatch: "
                f"{values} versus {reference_values}"
            )


def minmax_rescale_to_uint8(slice_2d):
    arr = np.asarray(slice_2d, dtype=np.float32)

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.uint8)

    vals = arr[finite_mask]
    lo = np.min(vals)
    hi = np.max(vals)

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    arr = (arr - lo) / (hi - lo + 1e-8)
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def resize_uint8_image(img_uint8, size):
    pil_img = Image.fromarray(img_uint8)
    pil_img = pil_img.resize((size, size), resample=Image.BILINEAR)
    return np.asarray(pil_img, dtype=np.uint8)


def save_gray_as_rgb_png(img_2d_uint8, out_png):
    safe_makedirs(os.path.dirname(out_png))
    img = Image.fromarray(img_2d_uint8.astype(np.uint8)).convert("RGB")
    img.save(out_png)


def save_mask_png(mask_slice_binary, out_png):
    safe_makedirs(os.path.dirname(out_png))
    disp = (mask_slice_binary > 0).astype(np.uint8) * 255
    img = Image.fromarray(disp).convert("RGB")
    img.save(out_png)


def save_rgb_png(img_rgb_uint8, out_png):
    safe_makedirs(os.path.dirname(out_png))
    img = Image.fromarray(img_rgb_uint8.astype(np.uint8)).convert("RGB")
    img.save(out_png)


def get_component_bbox(component_mask_binary):
    ys, xs = np.where(component_mask_binary > 0)

    if len(ys) == 0 or len(xs) == 0:
        return None

    ymin = int(np.min(ys))
    ymax = int(np.max(ys))
    xmin = int(np.min(xs))
    xmax = int(np.max(xs))
    area = int(np.sum(component_mask_binary > 0))

    return {
        "ymin": ymin,
        "ymax": ymax,
        "xmin": xmin,
        "xmax": xmax,
        "area": area,
    }


def get_component_stats(mask_slice_binary, score_slice):
    mask = (mask_slice_binary > 0).astype(np.uint8)

    if np.sum(mask) == 0:
        return []

    labeled, num_features = ndimage.label(mask)

    components = []

    for lab in range(1, num_features + 1):
        comp_mask = labeled == lab
        comp_area = int(np.sum(comp_mask))

        if comp_area == 0:
            continue

        comp_scores = score_slice[comp_mask]
        if comp_scores.size == 0:
            continue

        mean_prob = float(np.mean(comp_scores))
        bbox = get_component_bbox(comp_mask.astype(np.uint8))

        if bbox is None:
            continue

        score = mean_prob * math.sqrt(comp_area)

        components.append({
            "label": lab,
            "mask": comp_mask.astype(np.uint8),
            "area": comp_area,
            "mean_probability": mean_prob,
            "bbox": bbox,
            "score_meanprob_x_sqrtarea": score,
        })

    return components


def filter_components_by_min_size(components, min_component_pixels=MIN_COMPONENT_PIXELS):
    filtered = [c for c in components if c["area"] >= min_component_pixels]
    if len(filtered) > 0:
        return filtered
    return components


def select_best_component_meanprob_sqrtarea(components):
    if len(components) == 0:
        return None
    return max(components, key=lambda c: c["score_meanprob_x_sqrtarea"])


SPECIAL_SLICE_RULES = {}
SLICE_RULES_PATH = os.environ.get("DCE_SLICE_RULES_JSON", "")
if SLICE_RULES_PATH:
    with open(SLICE_RULES_PATH, "r", encoding="utf-8") as handle:
        SPECIAL_SLICE_RULES = json.load(handle)

DEFAULT_RULE = "reverse"
ALLOWED_RULES = {"pacs", "pacs_minus_1", "reverse", "reverse_minus_1"}


def clamp_index(idx, n_slices):
    return max(0, min(int(idx), n_slices - 1))


def get_rule_for_subject(subject_id):
    rule = SPECIAL_SLICE_RULES.get(subject_id, DEFAULT_RULE)
    if rule not in ALLOWED_RULES:
        raise ValueError(f"Unknown slice rule '{rule}' for subject {subject_id}")
    return rule


def choose_final_index(best_slice_pacs, n_slices, rule):
    if rule == "pacs":
        idx = best_slice_pacs
    elif rule == "pacs_minus_1":
        idx = best_slice_pacs - 1
    elif rule == "reverse":
        idx = n_slices - best_slice_pacs
    elif rule == "reverse_minus_1":
        idx = n_slices - best_slice_pacs - 1
    else:
        raise ValueError(f"Unsupported rule: {rule}")

    return clamp_index(idx, n_slices)


def map_raw_slice_to_resampled_slice(raw_img, resampled_img, raw_z):
    """
    Convert doctor-guided raw slice index to corresponding 1mm resampled slice index.
    """
    raw_size = raw_img.GetSize()  # x, y, z

    raw_x = raw_size[0] // 2
    raw_y = raw_size[1] // 2
    raw_index = (raw_x, raw_y, int(raw_z))

    physical_point = raw_img.TransformIndexToPhysicalPoint(raw_index)
    resampled_index = resampled_img.TransformPhysicalPointToIndex(physical_point)

    resampled_z = resampled_index[2]
    return clamp_index(resampled_z, resampled_img.GetSize()[2])


def select_best_component_near_mapped_slice(mask_vol, score_vol, mapped_z, window=2):
    """
    Doctor-guided selection:
    search only mapped_z ± window, not entire volume.
    """
    best = None

    z_start = max(0, mapped_z - window)
    z_end = min(mask_vol.shape[0], mapped_z + window + 1)

    for z in range(z_start, z_end):
        mask_slice = (mask_vol[z] > 0).astype(np.uint8)

        if np.sum(mask_slice) == 0:
            continue

        score_slice = score_vol[z].astype(np.float32)

        components = get_component_stats(mask_slice, score_slice)
        components = filter_components_by_min_size(
            components,
            min_component_pixels=MIN_COMPONENT_PIXELS
        )

        comp = select_best_component_meanprob_sqrtarea(components)

        if comp is None:
            continue

        if best is None or comp["score_meanprob_x_sqrtarea"] > best["component"]["score_meanprob_x_sqrtarea"]:
            best = {
                "z": z,
                "component": comp,
                "num_components": len(components),
                "mapped_z": mapped_z,
                "search_window": window,
            }

    return best


def build_bbox_overlay_rgb(whole_slice_uint8, bbox, thickness=BBOX_THICKNESS):
    out = np.stack(
        [whole_slice_uint8, whole_slice_uint8, whole_slice_uint8],
        axis=-1
    ).astype(np.uint8)

    if bbox is None:
        return out

    ymin = bbox["ymin"]
    ymax = bbox["ymax"]
    xmin = bbox["xmin"]
    xmax = bbox["xmax"]

    h, w = out.shape[:2]

    for t in range(thickness):
        y_top = max(0, ymin - t)
        y_bottom = min(h - 1, ymax + t)
        x_left = max(0, xmin - t)
        x_right = min(w - 1, xmax + t)

        out[y_top, x_left:x_right + 1, :] = [255, 255, 0]
        out[y_bottom, x_left:x_right + 1, :] = [255, 255, 0]
        out[y_top:y_bottom + 1, x_left, :] = [255, 255, 0]
        out[y_top:y_bottom + 1, x_right, :] = [255, 255, 0]

    return out


def crop_bbox_from_slice(slice_2d, bbox):
    if bbox is None:
        return None

    ymin = bbox["ymin"]
    ymax = bbox["ymax"]
    xmin = bbox["xmin"]
    xmax = bbox["xmax"]

    return slice_2d[ymin:ymax + 1, xmin:xmax + 1]


def main():
    required_paths = {
        "DCE_METADATA_XLSX": EXCEL_PATH,
        "DCE_NIFTI_ROOT": NIFTI_ROOT,
        "DCE_SEGMENTATION_ROOT": SEGMENTATION_ROOT,
        "DCE_STEP5_OUTPUT": OUTPUT_DIR,
    }
    missing = [name for name, value in required_paths.items() if not value]
    if missing:
        raise RuntimeError("Missing required environment variables: " + ", ".join(missing))

    safe_makedirs(OUTPUT_DIR)

    selected_mask_png_dir = os.path.join(OUTPUT_DIR, "selected_mask_pngs_resampled_1mm")
    bbox_overlay_png_dir = os.path.join(OUTPUT_DIR, "bbox_overlay_pngs_BCR_preprocessed")
    crop_224_png_dir = os.path.join(OUTPUT_DIR, "bbox_crop_224x224_pngs_BCR_N4_1mm_minmax")

    bias_corrected_nifti_dir = os.path.join(OUTPUT_DIR, "debug_bias_corrected_nifti")
    pre_resampled_nifti_dir = os.path.join(OUTPUT_DIR, "debug_pre_BCR_N4_1mm_nifti")
    po1_resampled_nifti_dir = os.path.join(OUTPUT_DIR, "debug_po1_BCR_N4_1mm_nifti")
    po2_resampled_nifti_dir = os.path.join(OUTPUT_DIR, "debug_po2_BCR_N4_1mm_nifti")
    prob_resampled_nifti_dir = os.path.join(OUTPUT_DIR, "debug_prob_resampled_to_BCR_1mm_nifti")
    task_log_dir = os.path.join(OUTPUT_DIR, "task_logs")

    safe_makedirs(selected_mask_png_dir)
    safe_makedirs(bbox_overlay_png_dir)
    safe_makedirs(crop_224_png_dir)
    safe_makedirs(bias_corrected_nifti_dir)
    safe_makedirs(pre_resampled_nifti_dir)
    safe_makedirs(po1_resampled_nifti_dir)
    safe_makedirs(po2_resampled_nifti_dir)
    safe_makedirs(prob_resampled_nifti_dir)
    safe_makedirs(task_log_dir)

    if SAVE_NPY:
        selected_mask_npy_dir = os.path.join(OUTPUT_DIR, "selected_mask_npy_resampled_1mm")
        crop_raw_npy_dir = os.path.join(OUTPUT_DIR, "bbox_crop_raw_npy_BCR_N4_1mm")
        crop_224_npy_dir = os.path.join(OUTPUT_DIR, "bbox_crop_224x224_npy_BCR_N4_1mm_minmax")

        safe_makedirs(selected_mask_npy_dir)
        safe_makedirs(crop_raw_npy_dir)
        safe_makedirs(crop_224_npy_dir)
    else:
        selected_mask_npy_dir = None
        crop_raw_npy_dir = None
        crop_224_npy_dir = None

    df = pd.read_excel(EXCEL_PATH, sheet_name=EXCEL_SHEET_NAME, engine="openpyxl")
    df.columns = df.columns.astype(str).str.strip()
    
    
    required_cols = [BEST_SLICE_COL, PRIMARY_SUBJECT_COL, SECONDARY_SUBJECT_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # One SLURM array task processes exactly one Excel row/subject.
    task_id_text = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id_text is None:
        raise RuntimeError(
            "SLURM_ARRAY_TASK_ID is not set. Submit this script as a SLURM array job."
        )

    task_id = int(task_id_text)
    total_rows = len(df)

    if task_id < 0 or task_id >= total_rows:
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={task_id} is outside the Excel row range "
            f"0-{total_rows - 1}."
        )

    original_excel_index = task_id
    df = df.iloc[[task_id]].copy()

    print(
        f"SLURM array task {task_id}: processing exactly one Excel row "
        f"out of {total_rows}"
    )

    records = []

    for _, row in df.iterrows():
        primary_sid = normalize_subject_id(row.get(PRIMARY_SUBJECT_COL, None))
        secondary_sid = normalize_subject_id(row.get(SECONDARY_SUBJECT_COL, None))
        best_slice_raw = row.get(BEST_SLICE_COL, None)

        subject_id = primary_sid
        subject_source_col = PRIMARY_SUBJECT_COL

        if secondary_sid in SPECIAL_USE_SECONDARY_ONLY:
            subject_id = secondary_sid
            subject_source_col = SECONDARY_SUBJECT_COL

        rec = {
            "excel_row": original_excel_index + 2,
            "slurm_array_task_id": task_id,
            "subject_id": subject_id,
            "subject_source_col": subject_source_col,
            "primary_sid": primary_sid,
            "secondary_sid": secondary_sid,
            "best_slice_raw": best_slice_raw,
            "best_slice_pacs_original": None,
            "whole_pre_path": None,
            "whole_po1_path": None,
            "whole_po2_path": None,
            "whole_case_type": None,
            "seg_subject_dir": None,
            "seg_folder_source": None,
            "prob_path": None,
            "original_po1_size_xyz": None,
            "bcr_po1_size_xyz": None,
            "original_po1_spacing_xyz": None,
            "bcr_po1_spacing_xyz": None,
            "original_prob_size_xyz": None,
            "resampled_prob_size_xyz": None,
            "selected_resampled_z_idx": None,
            "num_components_used": None,
            "selected_area": None,
            "selected_mean_probability": None,
            "selected_score_meanprob_x_sqrtarea": None,
            "bbox_ymin": None,
            "bbox_ymax": None,
            "bbox_xmin": None,
            "bbox_xmax": None,
            "crop_height": None,
            "crop_width": None,
            "selected_mask_png_path": None,
            "bbox_overlay_png_path": None,
            "crop_224_png_path_BCR_N4_1mm_minmax": None,
            "selected_mask_npy_path": None,
            "crop_raw_npy_path": None,
            "crop_224_npy_path_BCR_N4_1mm_minmax": None,
            "pre_bias_corrected_nifti_path": None,
            "po1_bias_corrected_nifti_path": None,
            "po2_bias_corrected_nifti_path": None,
            "pre_resampled_nifti_path": None,
            "po1_resampled_nifti_path": None,
            "po2_resampled_nifti_path": None,
            "prob_resampled_nifti_path": None,
            "status": None,
            "reason": None,
        }

        if subject_id is None:
            rec["status"] = "SKIP"
            rec["reason"] = "No usable subject ID"
            records.append(rec)
            continue

        rec["best_slice_pacs_original"] = parse_best_slice(best_slice_raw)

        phase_paths, missing_phases = find_phase_paths(subject_id)
        if missing_phases:
            rec["status"] = "SKIP"
            rec["reason"] = (
                "Missing phase files in combined Steps 1+2 output: "
                + ", ".join(missing_phases)
            )
            records.append(rec)
            continue

        rec["whole_pre_path"] = phase_paths["pre"]
        rec["whole_po1_path"] = phase_paths["po1"]
        rec["whole_po2_path"] = phase_paths["po2"]
        rec["whole_case_type"] = "combined_step1_step2"

        seg_subject_dir, seg_folder_source = find_subject_segmentation_folder(subject_id)
        if seg_subject_dir is None:
            rec["status"] = "SKIP"
            rec["reason"] = "Subject folder not found in combined Steps 3+4 output"
            records.append(rec)
            continue

        rec["seg_subject_dir"] = seg_subject_dir
        rec["seg_folder_source"] = seg_folder_source

        try:
            prob_path = find_probability_path(seg_subject_dir)
            rec["prob_path"] = prob_path

            pre_raw_img = sitk.ReadImage(phase_paths["pre"], sitk.sitkFloat32)
            po1_raw_img = sitk.ReadImage(phase_paths["po1"], sitk.sitkFloat32)
            po2_raw_img = sitk.ReadImage(phase_paths["po2"], sitk.sitkFloat32)
            prob_raw_img = sitk.ReadImage(prob_path, sitk.sitkFloat32)
            
            rec["original_po1_size_xyz"] = str(po1_raw_img.GetSize())
            rec["original_po1_spacing_xyz"] = str(po1_raw_img.GetSpacing())
            rec["original_prob_size_xyz"] = str(prob_raw_img.GetSize())
            
            # 1. N4-correct PRE, PO1 and PO2. PO1 defines the 1-mm
            # reference grid; PRE, PO2 and the probability map are
            # resampled into that exact grid.
            pre_bias_corrected_path = os.path.join(
                bias_corrected_nifti_dir,
                f"{subject_id}_pre_N4_corrected.nii.gz"
            )

            po1_bias_corrected_path = os.path.join(
                bias_corrected_nifti_dir,
                f"{subject_id}_po1_N4_corrected.nii.gz"
            )

            po2_bias_corrected_path = os.path.join(
                bias_corrected_nifti_dir,
                f"{subject_id}_po2_N4_corrected.nii.gz"
            )

            pre_resampled_path = os.path.join(
                pre_resampled_nifti_dir,
                f"{subject_id}_pre_N4_1mm.nii.gz"
            )

            po1_resampled_path = os.path.join(
                po1_resampled_nifti_dir,
                f"{subject_id}_po1_N4_1mm.nii.gz"
            )

            po2_resampled_path = os.path.join(
                po2_resampled_nifti_dir,
                f"{subject_id}_po2_N4_1mm.nii.gz"
            )

            prob_resampled_path = os.path.join(
                prob_resampled_nifti_dir,
                f"{subject_id}_prob_resampled_to_po1_N4_1mm.nii.gz"
            )

            final_paths = [
                pre_resampled_path,
                po1_resampled_path,
                po2_resampled_path,
                prob_resampled_path,
            ]

            print("\nDEBUG PATH CHECK")
            for output_path in final_paths:
                print(output_path, "exists =", os.path.exists(output_path))

            if all(os.path.exists(path) for path in final_paths):
                print(
                    f"[REUSE] {subject_id}: using existing N4+1mm "
                    "PRE/PO1/PO2 and probability map"
                )

                pre_bcr_img = sitk.ReadImage(pre_resampled_path, sitk.sitkFloat32)
                po1_bcr_img = sitk.ReadImage(po1_resampled_path, sitk.sitkFloat32)
                po2_bcr_img = sitk.ReadImage(po2_resampled_path, sitk.sitkFloat32)
                prob_resampled_img = sitk.ReadImage(prob_resampled_path, sitk.sitkFloat32)

            else:
                print(
                    f"[CREATE] {subject_id}: running PRE/PO1/PO2 "
                    "N4 correction and common-grid 1-mm resampling"
                )

                pre_corrected_img = n4_bias_field_correction(pre_raw_img)
                po1_corrected_img = n4_bias_field_correction(po1_raw_img)
                po2_corrected_img = n4_bias_field_correction(po2_raw_img)

                sitk.WriteImage(pre_corrected_img, pre_bias_corrected_path)
                sitk.WriteImage(po1_corrected_img, po1_bias_corrected_path)
                sitk.WriteImage(po2_corrected_img, po2_bias_corrected_path)

                po1_bcr_img, _, _, _, _ = resample_image_to_1mm(
                    po1_corrected_img,
                    is_label=False
                )

                pre_bcr_img = resample_image_to_reference(
                    pre_corrected_img,
                    po1_bcr_img
                )

                po2_bcr_img = resample_image_to_reference(
                    po2_corrected_img,
                    po1_bcr_img
                )

                sitk.WriteImage(pre_bcr_img, pre_resampled_path)
                sitk.WriteImage(po1_bcr_img, po1_resampled_path)
                sitk.WriteImage(po2_bcr_img, po2_resampled_path)

                prob_resampled_img = resample_probability_to_reference(
                    prob_raw_img,
                    po1_bcr_img
                )
                sitk.WriteImage(prob_resampled_img, prob_resampled_path)

            assert_same_geometry(pre_bcr_img, po1_bcr_img, "PRE N4/1mm")
            assert_same_geometry(po2_bcr_img, po1_bcr_img, "PO2 N4/1mm")
            assert_same_geometry(
                prob_resampled_img,
                po1_bcr_img,
                "Probability map in PO1 N4/1mm space"
            )

            rec["pre_bias_corrected_nifti_path"] = pre_bias_corrected_path
            rec["po1_bias_corrected_nifti_path"] = po1_bias_corrected_path
            rec["po2_bias_corrected_nifti_path"] = po2_bias_corrected_path
            rec["pre_resampled_nifti_path"] = pre_resampled_path
            rec["po1_resampled_nifti_path"] = po1_resampled_path
            rec["po2_resampled_nifti_path"] = po2_resampled_path
            rec["prob_resampled_nifti_path"] = prob_resampled_path
            
            rec["bcr_po1_size_xyz"] = str(po1_bcr_img.GetSize())
            rec["bcr_po1_spacing_xyz"] = str(po1_bcr_img.GetSpacing())
            rec["resampled_prob_size_xyz"] = str(prob_resampled_img.GetSize())
            po1_bcr_vol = sitk.GetArrayFromImage(po1_bcr_img).astype(np.float32)
            prob_resampled_vol = sitk.GetArrayFromImage(prob_resampled_img).astype(np.float32)

            if po1_bcr_vol.ndim != 3:
                rec["status"] = "FAIL"
                rec["reason"] = f"BCR PO1 volume is not 3D, shape={po1_bcr_vol.shape}"
                records.append(rec)
                continue

            if prob_resampled_vol.ndim != 3:
                rec["status"] = "FAIL"
                rec["reason"] = f"Resampled probability volume is not 3D, shape={prob_resampled_vol.shape}"
                records.append(rec)
                continue

            if po1_bcr_vol.shape != prob_resampled_vol.shape:
                rec["status"] = "FAIL"
                rec["reason"] = (
                    f"Shape mismatch after resampling: "
                    f"PO1={po1_bcr_vol.shape}, prob={prob_resampled_vol.shape}"
                )
                records.append(rec)
                continue

            mask_resampled_vol = (prob_resampled_vol > PROB_THRESHOLD).astype(np.uint8)
# 4. Use doctor-guided slice mapping, then select best component near mapped slice
            
            best_slice_pacs = parse_best_slice(best_slice_raw)

            if best_slice_pacs is None:
                rec["status"] = "SKIP"
                rec["reason"] = f"Best slice unavailable: {best_slice_raw}"
                records.append(rec)
                continue
            
            raw_n_slices = po1_raw_img.GetSize()[2]
            
            rule = get_rule_for_subject(subject_id)
            raw_chosen_idx = choose_final_index(best_slice_pacs, raw_n_slices, rule)
            
            mapped_z = map_raw_slice_to_resampled_slice(
                raw_img=po1_raw_img,
                resampled_img=po1_bcr_img,
                raw_z=raw_chosen_idx
            )
            
            best = select_best_component_near_mapped_slice(
                mask_resampled_vol,
                prob_resampled_vol,
                mapped_z=mapped_z,
                window=2
            )
            
            rec["best_slice_pacs_original"] = best_slice_pacs
            rec["raw_chosen_idx_from_old_logic"] = raw_chosen_idx
            rec["mapped_resampled_z_idx"] = mapped_z
            rec["rule_used"] = rule

            if best is None:
                rec["status"] = "FAIL"
                rec["reason"] = "No tumor component found after resampling probability map to BCR 1mm space"
                records.append(rec)
                continue

            z = best["z"]
            comp = best["component"]
            selected_mask_binary = comp["mask"].astype(np.uint8)
            bbox = comp["bbox"]

            rec["selected_resampled_z_idx"] = z
            rec["num_components_used"] = best["num_components"]
            rec["selected_area"] = comp["area"]
            rec["selected_mean_probability"] = comp["mean_probability"]
            rec["selected_score_meanprob_x_sqrtarea"] = comp["score_meanprob_x_sqrtarea"]

            rec["bbox_ymin"] = bbox["ymin"]
            rec["bbox_ymax"] = bbox["ymax"]
            rec["bbox_xmin"] = bbox["xmin"]
            rec["bbox_xmax"] = bbox["xmax"]

            # 5. Crop from BCR-preprocessed image, not raw image
            po1_slice_bcr = po1_bcr_vol[z]
            crop_raw_bcr = crop_bbox_from_slice(po1_slice_bcr, bbox)

            if crop_raw_bcr is None or crop_raw_bcr.size == 0:
                rec["status"] = "FAIL"
                rec["reason"] = "Empty crop from selected BCR-space bounding box"
                records.append(rec)
                continue

            rec["crop_height"] = crop_raw_bcr.shape[0]
            rec["crop_width"] = crop_raw_bcr.shape[1]

            # 6. Match training-style min-max normalization while saving
            crop_uint8_minmax = minmax_rescale_to_uint8(crop_raw_bcr)
            crop_224_uint8_minmax = resize_uint8_image(crop_uint8_minmax, 224)

            # for overlay visualization only
            whole_slice_uint8 = minmax_rescale_to_uint8(po1_slice_bcr)
            bbox_overlay_rgb = build_bbox_overlay_rgb(
                whole_slice_uint8,
                bbox,
                thickness=BBOX_THICKNESS
            )

            selected_mask_png_path = os.path.join(
                selected_mask_png_dir,
                f"{subject_id}_z{z:03d}_selected_mask_BCR_1mm.png"
            )

            bbox_overlay_png_path = os.path.join(
                bbox_overlay_png_dir,
                f"{subject_id}_z{z:03d}_bbox_overlay_BCR_1mm.png"
            )

            crop_224_png_path = os.path.join(
                crop_224_png_dir,
                f"{subject_id}_z{z:03d}_bbox_crop_224x224_BCR_N4_1mm_minmax.png"
            )

            save_mask_png(selected_mask_binary, selected_mask_png_path)
            save_rgb_png(bbox_overlay_rgb, bbox_overlay_png_path)
            save_gray_as_rgb_png(crop_224_uint8_minmax, crop_224_png_path)

            rec["selected_mask_png_path"] = selected_mask_png_path
            rec["bbox_overlay_png_path"] = bbox_overlay_png_path
            rec["crop_224_png_path_BCR_N4_1mm_minmax"] = crop_224_png_path

            if SAVE_NPY:
                selected_mask_npy_path = os.path.join(
                    selected_mask_npy_dir,
                    f"{subject_id}_z{z:03d}_selected_mask_BCR_1mm.npy"
                )

                crop_raw_npy_path = os.path.join(
                    crop_raw_npy_dir,
                    f"{subject_id}_z{z:03d}_bbox_crop_raw_BCR_N4_1mm.npy"
                )

                crop_224_npy_path = os.path.join(
                    crop_224_npy_dir,
                    f"{subject_id}_z{z:03d}_bbox_crop_224x224_BCR_N4_1mm_minmax.npy"
                )

                np.save(selected_mask_npy_path, selected_mask_binary)
                np.save(crop_raw_npy_path, crop_raw_bcr)
                np.save(crop_224_npy_path, crop_224_uint8_minmax)

                rec["selected_mask_npy_path"] = selected_mask_npy_path
                rec["crop_raw_npy_path"] = crop_raw_npy_path
                rec["crop_224_npy_path_BCR_N4_1mm_minmax"] = crop_224_npy_path

            rec["status"] = "OK"
            rec["reason"] = ""

        except Exception as e:
            rec["status"] = "FAIL"
            rec["reason"] = str(e)

        records.append(rec)

    log_df = pd.DataFrame(records)

    subject_for_filename = "no_subject"
    if records and records[0].get("subject_id"):
        subject_for_filename = re.sub(
            r"[^A-Za-z0-9_.-]+",
            "_",
            str(records[0]["subject_id"])
        )

    csv_path = os.path.join(
        task_log_dir,
        f"step5_task_{task_id:04d}_{subject_for_filename}.csv"
    )
    log_df.to_csv(csv_path, index=False)

    n_ok = int((log_df["status"] == "OK").sum())
    n_skip = int((log_df["status"] == "SKIP").sum())
    n_fail = int((log_df["status"] == "FAIL").sum())

    print("\nDone.")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"BCR-aligned crop PNG dir: {crop_224_png_dir}")
    print(f"BBox overlay PNG dir: {bbox_overlay_png_dir}")
    print(f"Selected mask PNG dir: {selected_mask_png_dir}")
    print(f"CSV log: {csv_path}")
    print(f"OK={n_ok}, SKIP={n_skip}, FAIL={n_fail}")

    if SAVE_NPY:
        print("Saved NPY outputs too.")


if __name__ == "__main__":
    main()
