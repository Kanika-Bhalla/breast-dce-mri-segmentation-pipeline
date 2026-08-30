#!/usr/bin/env python3
"""Run breast and tumor segmentation for one case or a directory of cases."""

import argparse
import copy
import os
import torch

from segmentation.ImageProcessing import Norm_Zscore, imgnorm, labeling_seg, read_image, save_image
from segmentation.Models_3D import ModelBreast, ModelTumor
from segmentation.Segmenation import BreastSeg, BreastTumor


def run_one_case(options, model_breast, model_tumor):
    os.makedirs(options.outfolder, exist_ok=True)
    print(f"PRE : {options.pre}\nPOST: {options.post}\nOUT : {options.outfolder}")
    image_pre, image_post, image_sub, spacing = read_image(options)
    image_pre = Norm_Zscore(imgnorm(image_pre))
    image_post = Norm_Zscore(imgnorm(image_post))
    image_sub = Norm_Zscore(imgnorm(image_sub))
    breast_mask = BreastSeg(image_pre, spacing, model_breast, options)
    probability, tumor_segmentation = BreastTumor(
        image_sub, image_post, breast_mask, spacing, model_tumor, model_tumor, options
    )
    save_image(breast_mask, options, "breast_mask.nii.gz")
    save_image(probability, options, "prob_1st.nii.gz")
    save_image(labeling_seg(tumor_segmentation), options, "seg_2nd.nii.gz")


def main():
    parser = argparse.ArgumentParser(description="DCE-MRI tumor segmentation")
    parser.add_argument("--cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument("--pre", help="Single-case pre.nii.gz")
    parser.add_argument("--post", help="Single-case po1.nii.gz")
    parser.add_argument("--data-root", default="", help="Directory containing subject folders")
    parser.add_argument("--outfolder", required=True)
    args = parser.parse_args()
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --cuda 0")
    model_breast = ModelBreast(1, 1)
    model_tumor = ModelTumor(3, 1)
    if args.cuda:
        model_breast, model_tumor = model_breast.cuda(), model_tumor.cuda()
    if args.data_root:
        subjects = sorted(name for name in os.listdir(args.data_root)
                          if os.path.isdir(os.path.join(args.data_root, name)))
        success = failed = 0
        for subject in subjects:
            pre_path = os.path.join(args.data_root, subject, "pre.nii.gz")
            post_path = os.path.join(args.data_root, subject, "po1.nii.gz")
            if not (os.path.exists(pre_path) and os.path.exists(post_path)):
                print(f"[SKIP] {subject}: missing pre.nii.gz or po1.nii.gz")
                failed += 1
                continue
            case = copy.copy(args)
            case.pre, case.post = pre_path, post_path
            case.outfolder = os.path.join(args.outfolder, subject)
            try:
                run_one_case(case, model_breast, model_tumor)
                success += 1
            except Exception as exc:
                print(f"[FAIL] {subject}: {exc}")
                failed += 1
        print(f"Batch finished. Success={success}, Failed/Skipped={failed}")
    else:
        if not args.pre or not args.post:
            parser.error("provide --pre and --post, or provide --data-root")
        run_one_case(args, model_breast, model_tumor)


if __name__ == "__main__":
    main()
