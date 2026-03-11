# nnMamba Cross-Domain Pipeline: ABUS → Duying

Train nnMamba on TDSC-ABUS (full supervision), then perform direct cross-domain inference on Duying (bbox-only labels), and evaluate predicted masks against GT bounding boxes.

## Pipeline Overview

| Step | Script | Where | Time Est. | Description |
|------|--------|-------|-----------|-------------|
| 1 | `data_conversion/convert_abus_to_nnmamba.py` | Local | ~1-2h | Convert 200 ABUS `.nrrd` → nnMamba `.nii.gz` |
| 2 | `training/train_nnmamba.sh` | HPC (GPU) | ~24-48h | Plan, preprocess, train nnMamba on ABUS |
| 3 | `inference/preprocess_duying.py` | Local | ~2-4h | Resample 316 Duying volumes 1×3×1→1×1×1mm + normalize |
| 4 | `inference/run_inference.sh` | HPC (GPU) | ~4-8h | nnMamba inference on Duying + resample preds back |
| 5 | `evaluation/evaluate_bbox.py` | HPC (CPU) | ~10min | 3D IoU / recall / precision vs GT bboxes |
| 6 | `visualization/visualize_failures.py` | HPC (CPU) | ~5min | Multi-panel figures for 10 worst cases |

```
LOCAL:  Step 1 ──┐    ┌──→ HPC:  Step 2 ──→ Step 4 ──→ Step 5 ──→ Step 6
LOCAL:  Step 3 ──┤    │                        ↑
                 └─ rsync to HPC ──────────────┘
(Steps 1 and 3 can run in parallel on your local machine)
```

## Prerequisites

### Conda Environment (both local and HPC)

```bash
conda create -n nnmamba python=3.10 -y
conda activate nnmamba

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nibabel numpy scipy pandas matplotlib pynrrd

# HPC only (needs CUDA):
pip install mamba-ssm causal-conv1d

# Install nnMamba (from repo root)
cd /path/to/nnMamba
pip install -e .
```

> On your local Mac, skip `mamba-ssm` and `causal-conv1d` — they require CUDA. Steps 1 and 3 don't need them.

---

## Part 1 — Local Data Processing

### Step 1: Convert ABUS to nnMamba Format

```bash
cd /Users/langtian/Desktop/NYU/MS\ Thesis/nnMamba/project

python data_conversion/convert_abus_to_nnmamba.py \
    --abus_root "/Volumes/Autzoko/Dataset/US43K/ABUS/data" \
    --output_base "/Users/langtian/Desktop/NYU/MS Thesis/nnMamba/project/local_output/nnMamba_raw/nnUNet_raw_data"
```

This produces:

```
local_output/nnMamba_raw/nnUNet_raw_data/Task001_ABUS/
├── imagesTr/       ABUS_XXX_0000.nii.gz   (200 volumes)
├── labelsTr/       ABUS_XXX.nii.gz        (200 masks, binary)
├── imagesTs/       (empty)
├── dataset.json
└── splits_final.pkl
```

The original ABUS split is preserved in `splits_final.pkl`:
- **Train**: 170 cases (original Train 100 + Test 70)
- **Val**: 30 cases (original Validation 30)

Test cases are folded into training since our real evaluation target is Duying (cross-domain).

### Step 3: Preprocess Duying Volumes

Can run simultaneously with Step 1 in another terminal:

```bash
python inference/preprocess_duying.py \
    --duying_root "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying" \
    --output_dir "/Users/langtian/Desktop/NYU/MS Thesis/nnMamba/project/local_output/duying_preprocessed" \
    --target_spacing 1.0 1.0 1.0 \
    --num_workers 8
```

For each volume: reads original spacing from NIfTI header (1×3×1mm), resamples to 1×1×1mm with trilinear interpolation, clips intensity at 0.5/99.5 percentile, z-score normalizes.

### Transfer to HPC

```bash
# Transfer converted ABUS data
rsync -avP local_output/nnMamba_raw/ \
    <user>@jubail.abudhabi.nyu.edu:/scratch/<user>/nnmamba_pipeline/nnMamba_raw/

# Transfer preprocessed Duying volumes
rsync -avP local_output/duying_preprocessed/ \
    <user>@jubail.abudhabi.nyu.edu:/scratch/<user>/nnmamba_pipeline/duying_preprocessed/

# Transfer the Duying root (needed for GT bboxes + postprocessing headers)
rsync -avP "/Volumes/Lang/Research/Data/3D Ultrasound/nnDet/Duying/" \
    <user>@jubail.abudhabi.nyu.edu:/scratch/<user>/data/Duying/

# Transfer project scripts
rsync -avP /Users/langtian/Desktop/NYU/MS\ Thesis/nnMamba/project/ \
    <user>@jubail.abudhabi.nyu.edu:/scratch/<user>/nnmamba_pipeline/project/
```

---

## Part 2 — HPC (Training + Inference + Evaluation)

SSH into Jubail and set up paths:

```bash
ssh <user>@jubail.abudhabi.nyu.edu
conda activate nnmamba

# Define paths (add to ~/.bashrc for persistence)
export WORK=/scratch/$USER/nnmamba_pipeline
export PROJECT=$WORK/project
```

### Step 2: nnMamba Training

```bash
# Set nnUNet environment variables
export nnUNet_raw_data_base="$WORK/nnMamba_raw"
export nnUNet_preprocessed="$WORK/nnMamba_preprocessed"
export RESULTS_FOLDER="$WORK/nnMamba_results"
export CUDA_VISIBLE_DEVICES=0

# 2a. Plan and preprocess
python -m nnunet.experiment_planning.nnUNet_plan_and_preprocess \
    -t 1 \
    --verify_dataset_integrity \
    -tf 8 -tl 8

# 2b. Copy splits_final.pkl into preprocessed directory
cp "$nnUNet_raw_data_base/nnUNet_raw_data/Task001_ABUS/splits_final.pkl" \
   "$nnUNet_preprocessed/Task001_ABUS/splits_final.pkl"

# 2c. Train (fold 0, ~1000 epochs)
python -m nnunet.run.run_training \
    3d_fullres \
    nnUNetTrainerV2_fullGIL_mamba \
    1 \
    0 \
    --npz
```

- 1000 epochs, LR 1e-4, poly schedule, FP16 mixed precision
- Model checkpoints saved to `$RESULTS_FOLDER/nnUNet/3d_fullres/Task001_ABUS/nnUNetTrainerV2_fullGIL_mamba__nnUNetPlansv2.1/fold_0/`

### Step 4: nnMamba Inference on Duying

```bash
export nnUNet_raw_data_base="$WORK/nnMamba_raw"
export nnUNet_preprocessed="$WORK/nnMamba_preprocessed"
export RESULTS_FOLDER="$WORK/nnMamba_results"
export CUDA_VISIBLE_DEVICES=0

# 4a. Predict (outputs at 1×1×1mm isotropic)
python -m nnunet.inference.predict_simple \
    -i "$WORK/duying_preprocessed" \
    -o "$WORK/predictions_isotropic" \
    -t 1 \
    -tr nnUNetTrainerV2_fullGIL_mamba \
    -m 3d_fullres \
    -f 0

# 4b. Resample predictions back to original Duying spacing (1×3×1mm)
python $PROJECT/inference/postprocess_predictions.py \
    --pred_dir "$WORK/predictions_isotropic" \
    --duying_root /scratch/$USER/data/Duying \
    --output_dir "$WORK/predictions_original_spacing"
```

### Step 5: Evaluation

```bash
python $PROJECT/evaluation/evaluate_bbox.py \
    --pred_dir "$WORK/predictions_original_spacing" \
    --duying_root /scratch/$USER/data/Duying \
    --output_dir "$WORK/results"
```

Outputs:
- `results/per_lesion_results.csv` — per-lesion IoU, recall, precision, bbox coordinates
- `results/summary.json` — mean±std IoU/recall/precision, detection rates at IoU > 0.1/0.3/0.5

### Step 6: Visualization

```bash
python $PROJECT/visualization/visualize_failures.py \
    --results_csv "$WORK/results/per_lesion_results.csv" \
    --duying_root /scratch/$USER/data/Duying \
    --pred_dir "$WORK/predictions_original_spacing" \
    --output_dir "$WORK/visualization" \
    --n_worst 10
```

Each figure (2×3 panels):
- Top row: full axial/coronal/sagittal slices at GT bbox center — GT (green solid) + pred (orange dashed)
- Bottom row: zoomed crops around the lesion
- Title: volume ID, lesion ID, IoU, GT bbox size

### Retrieve Results

```bash
# From your local machine
rsync -avP <user>@jubail.abudhabi.nyu.edu:/scratch/<user>/nnmamba_pipeline/results/ \
    local_output/results/

rsync -avP <user>@jubail.abudhabi.nyu.edu:/scratch/<user>/nnmamba_pipeline/visualization/ \
    local_output/visualization/
```

---

## SLURM Alternative

If you prefer batch jobs instead of running interactively, sbatch scripts are in `slurm_jobs/`. Update the paths inside each `.sbatch` file, then:

```bash
mkdir -p logs

# Step 2 (GPU)
JOB2=$(sbatch --parsable slurm_jobs/step2_train.sbatch)

# Step 4 (GPU) — after training finishes
JOB4=$(sbatch --parsable --dependency=afterok:$JOB2 slurm_jobs/step4_inference.sbatch)

# Steps 5+6 (CPU)
sbatch --dependency=afterok:$JOB4 slurm_jobs/step5_6_eval_vis.sbatch
```

---

## Output Directory Structure

```
/scratch/$USER/nnmamba_pipeline/
├── nnMamba_raw/nnUNet_raw_data/
│   └── Task001_ABUS/               # From local Step 1
├── nnMamba_preprocessed/
│   └── Task001_ABUS/                  # Generated by Step 2
├── nnMamba_results/nnUNet/3d_fullres/Task001_ABUS/
│   └── nnUNetTrainerV2_fullGIL_mamba__nnUNetPlansv2.1/
│       └── fold_0/                    # Trained model
├── duying_preprocessed/               # From local Step 3
├── predictions_isotropic/             # Step 4a output (1×1×1mm)
├── predictions_original_spacing/      # Step 4b output (1×3×1mm)
├── results/
│   ├── per_lesion_results.csv         # Step 5
│   └── summary.json                   # Step 5
└── visualization/
    └── worst_XX_case_XXXXX_*.png      # Step 6 (×10 figures)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: nrrd` | `pip install pynrrd` (local, for Step 1) |
| `ModuleNotFoundError: mamba_ssm` | `pip install mamba-ssm causal-conv1d` (HPC only, needs CUDA) |
| `nnUNet_raw_data_base is not defined` | Set env vars before running — see Step 2 commands above |
| OOM during training | Reduce batch size in nnUNet plans or request a larger GPU |
| OOM during inference | nnMamba uses sliding window; try `--mem 256G` in sbatch |
| `splits_final.pkl` not found | Ensure the `cp` command in Step 2b ran successfully |
| NRRD read errors (Step 1) | `pip install --upgrade pynrrd` (need ≥ 0.4.0) |
| Predictions all zeros | Check that intensity normalization matches training — verify the plans pkl |
