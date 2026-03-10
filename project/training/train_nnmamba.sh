#!/bin/bash
# =============================================================================
# Step 2: nnMamba Training on ABUS
#
# This script sets up environment variables and runs:
#   1. nnMamba plan & preprocess
#   2. nnMamba training (3d_fullres, fold 0)
#
# Usage:
#   bash train_nnmamba.sh \
#       --data_base /path/to/nnMamba_raw_parent \
#       --preprocessed /path/to/nnMamba_preprocessed \
#       --results /path/to/nnMamba_results \
#       [--task_id 1] \
#       [--fold 0] \
#       [--gpus 0]
# =============================================================================

set -e

# Default values
TASK_ID=1
FOLD=0
GPUS="0"
DATA_BASE=""
PREPROCESSED=""
RESULTS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_base) DATA_BASE="$2"; shift 2 ;;
        --preprocessed) PREPROCESSED="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        --task_id) TASK_ID="$2"; shift 2 ;;
        --fold) FOLD="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_BASE" || -z "$PREPROCESSED" || -z "$RESULTS" ]]; then
    echo "Error: --data_base, --preprocessed, and --results are required"
    exit 1
fi

# Set nnUNet environment variables
export nnUNet_raw_data_base="$DATA_BASE"
export nnUNet_preprocessed="$PREPROCESSED"
export RESULTS_FOLDER="$RESULTS"
export CUDA_VISIBLE_DEVICES="$GPUS"

TASK_NAME="Task$(printf '%03d' $TASK_ID)_ABUS"

echo "============================================="
echo "nnMamba Training Pipeline"
echo "============================================="
echo "Raw data base:  $nnUNet_raw_data_base"
echo "Preprocessed:   $nnUNet_preprocessed"
echo "Results:        $RESULTS_FOLDER"
echo "Task:           $TASK_NAME (ID=$TASK_ID)"
echo "Fold:           $FOLD"
echo "GPUs:           $GPUS"
echo "============================================="

# --- Step 2a: Copy splits_final.pkl to preprocessed dir ---
# nnUNet expects splits_final.pkl in the preprocessed task folder.
# We created it during data conversion; copy it after preprocessing creates the dir.

# --- Step 2b: Plan and Preprocess ---
echo ""
echo ">>> Running nnMamba plan and preprocess..."
python -m nnunet.experiment_planning.nnUNet_plan_and_preprocess \
    -t $TASK_ID \
    --verify_dataset_integrity \
    -tf 8 \
    -tl 8

# Copy splits_final.pkl into preprocessed directory
SPLITS_SRC="$nnUNet_raw_data_base/nnUNet_raw_data/$TASK_NAME/splits_final.pkl"
SPLITS_DST="$nnUNet_preprocessed/$TASK_NAME/splits_final.pkl"
if [[ -f "$SPLITS_SRC" ]]; then
    echo ">>> Copying splits_final.pkl to preprocessed directory..."
    cp "$SPLITS_SRC" "$SPLITS_DST"
    echo "    Copied: $SPLITS_DST"
else
    echo "Warning: splits_final.pkl not found at $SPLITS_SRC"
    echo "  nnUNet will generate its own 5-fold split"
fi

# --- Step 2c: Train ---
echo ""
echo ">>> Starting nnMamba training (3d_fullres, fold $FOLD)..."
python -m nnunet.run.run_training \
    3d_fullres \
    nnUNetTrainerV2_fullGIL_mamba \
    $TASK_ID \
    $FOLD \
    --npz

echo ""
echo "============================================="
echo "Training complete!"
echo "Model saved to: $RESULTS_FOLDER/nnUNet/3d_fullres/$TASK_NAME/nnUNetTrainerV2_fullGIL_mamba__nnUNetPlansv2.1/"
echo "============================================="
