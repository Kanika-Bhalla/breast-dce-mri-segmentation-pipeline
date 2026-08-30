#!/bin/bash
#SBATCH --job-name=dce_step5
#SBATCH --partition=tier2_cpu
#SBATCH --mem=10G
#SBATCH --cpus-per-task=5
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%A_%a.stdout.txt
#SBATCH --error=logs/%x_%A_%a.stderr.txt

set -euo pipefail
: "${DCE_REPO:?Set DCE_REPO to the cloned repository}"
mkdir -p "$DCE_REPO/logs"
cd "$DCE_REPO"
python scripts/step5_multiphase_preprocess_and_select_tumor.py
