#!/bin/bash
#SBATCH --job-name=dce_step1_2
#SBATCH --partition=tier2_cpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.stdout.txt
#SBATCH --error=logs/%x_%A_%a.stderr.txt

set -euo pipefail
: "${DCE_REPO:?Set DCE_REPO to the cloned repository}"
mkdir -p "$DCE_REPO/logs"
cd "$DCE_REPO"
python scripts/step1_2_dicom_to_nifti.py "${SLURM_ARRAY_TASK_ID}"
