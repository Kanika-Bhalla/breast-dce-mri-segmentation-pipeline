#!/bin/bash
#SBATCH --job-name=dce_step3_4
#SBATCH --partition=tier2_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.stdout.txt
#SBATCH --error=logs/%x_%A_%a.stderr.txt

set -euo pipefail
: "${DCE_REPO:?Set DCE_REPO}"
: "${DCE_NIFTI_ROOT:?Set DCE_NIFTI_ROOT}"
: "${DCE_SEGMENTATION_ROOT:?Set DCE_SEGMENTATION_ROOT}"
mapfile -t subjects < <(find "$DCE_NIFTI_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
subject="${subjects[$SLURM_ARRAY_TASK_ID]}"
cd "$DCE_REPO"
python scripts/step3_4_segment.py --cuda 1 \
  --pre "$DCE_NIFTI_ROOT/$subject/pre.nii.gz" \
  --post "$DCE_NIFTI_ROOT/$subject/po1.nii.gz" \
  --outfolder "$DCE_SEGMENTATION_ROOT/$subject"
