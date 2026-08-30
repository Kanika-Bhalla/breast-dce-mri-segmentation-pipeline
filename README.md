# Breast DCE-MRI segmentation and multiphase MIP pipeline

Research pipeline for converting DCE-MRI DICOM series into aligned PRE, PO1
and PO2 NIfTI volumes, producing breast/tumor segmentations, applying N4 bias
correction and 1-mm alignment, and creating tumor-range multiphase MIPs.

## Privacy

**No patient data are included.** Do not commit DICOM, NIfTI, MIP, mask,
probability-map, spreadsheet, Word, log, checkpoint, or subject-specific
exception files. The `.gitignore` blocks these common formats, but every commit
must still be reviewed manually. All paths in this repository are placeholders
and must point to authorized storage outside the repository.

## Pipeline

1. `step1_2_dicom_to_nifti.py`: sort DICOM slices using DICOM metadata and
   export `pre.nii.gz`, `po1.nii.gz`, and `po2.nii.gz`. Optional stacked-series
   cases are supplied through a private JSON configuration.
2. `step3_4_segment.py`: normalize PRE, PO1, and their subtraction; generate
   `breast_mask.nii.gz`, `prob_1st.nii.gz`, and `seg_2nd.nii.gz`.
3. `step5_multiphase_preprocess_and_select_tumor.py`: N4-correct all three
   phases, define a 1-mm PO1 reference grid, align PRE/PO2/probability to PO1,
   and select the tumor component near the reviewed PO1 slice.
4. `step6_generate_multiphase_mips.py`: apply the same reviewed z range to the
   aligned PRE/PO1/PO2 volumes, take a maximum-intensity projection, scale each
   MIP by its 1st/99th percentiles, and resize to 224×224.

## Expected private inputs

The phase table used by Step 1 must contain `Session`, `Pre_Path`,
`Post_1_Path`, and `Post_2_Path`. Step 5 expects the configured metadata sheet
and columns shown at the top of its script. Step 6 expects a Word table with
`Subject ID` and `My range`. These files stay outside Git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp configs/paths.env.example paths.env
# Edit paths.env locally. It is ignored by Git.
source paths.env
export DCE_REPO="$PWD"
```

Place authorized segmentation checkpoints in `Models/` because the upstream
segmentation functions load them from that directory. See `Models/README.md`.

## SLURM submission

Do not hardcode an array ending such as `0-124`. Calculate it from the current
input table so later rows are not silently omitted.

```bash
# Step 1+2: number of phase-table rows
N=$(python -c 'import os,pandas as p; f=os.environ["DCE_PHASE_TABLE"]; print(len(p.read_excel(f) if f.endswith((".xlsx",".xls")) else p.read_csv(f)))')
sbatch --array="0-$((N-1))" slurm/step1_2_array.sh

# Step 3+4: number of subject directories
N=$(find "$DCE_NIFTI_ROOT" -mindepth 1 -maxdepth 1 -type d | wc -l)
sbatch --array="0-$((N-1))" slurm/step3_4_array.sh

# Step 5: number of metadata rows
N=$(python -c 'import os,pandas as p; print(len(p.read_excel(os.environ["DCE_METADATA_XLSX"],sheet_name="Data")))')
sbatch --array="0-$((N-1))" slurm/step5_array.sh

# Step 6: use the number of valid reviewed ranges, not the full workbook size.
N=$(python -c 'from scripts.step6_generate_multiphase_mips import read_manual_ranges; print(len(read_manual_ranges()))')
sbatch --array="0-$((N-1))" slurm/step6_array.sh
```

Review the job logs and verify matching geometry and counts before proceeding
to the next step.

## Segmentation attribution

The segmentation implementation is adapted from Jun Zhang/Mazurowski Lab's
[MRI breast tumor segmentation repository](https://github.com/MaciejMazurowski/mri-breast-tumor-segmentation).
See `THIRD_PARTY_NOTICE.md` and the upstream license. Model checkpoints are not
redistributed here.

## Intended use

Research use only. This pipeline is not a medical device and is not intended
for clinical diagnosis or treatment decisions.
