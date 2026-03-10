#!/bin/bash
# =============================================================================
# Step 4: nnMamba Inference on Duying
#
# Runs nnMamba prediction on preprocessed Duying volumes and then
# resamples predicted masks back to original Duying spacing (1x3x1mm).
#
# Usage:
#   bash run_inference.sh \
#       --input_dir /path/to/duying_preprocessed \
#       --output_dir /path/to/predictions_isotropic \
#       --final_dir /path/to/predictions_original_spacing \
#       --data_base /path/to/nnMamba_raw_parent \
#       --preprocessed /path/to/nnMamba_preprocessed \
#       --results /path/to/nnMamba_results \
#       --duying_root /path/to/Duying \
#       [--task_id 1] \
#       [--fold 0] \
#       [--gpus 0]
# =============================================================================

set -e

# Defaults
TASK_ID=1
FOLD=0
GPUS="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir) INPUT_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --final_dir) FINAL_DIR="$2"; shift 2 ;;
        --data_base) DATA_BASE="$2"; shift 2 ;;
        --preprocessed) PREPROCESSED="$2"; shift 2 ;;
        --results) RESULTS="$2"; shift 2 ;;
        --duying_root) DUYING_ROOT="$2"; shift 2 ;;
        --task_id) TASK_ID="$2"; shift 2 ;;
        --fold) FOLD="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export nnUNet_raw_data_base="$DATA_BASE"
export nnUNet_preprocessed="$PREPROCESSED"
export RESULTS_FOLDER="$RESULTS"
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "============================================="
echo "nnMamba Inference on Duying"
echo "============================================="
echo "Input (preprocessed):  $INPUT_DIR"
echo "Output (isotropic):    $OUTPUT_DIR"
echo "Final (orig spacing):  $FINAL_DIR"
echo "Task ID:               $TASK_ID"
echo "Fold:                  $FOLD"
echo "============================================="

# Step 4a: Run nnMamba prediction
echo ""
echo ">>> Running nnMamba prediction..."
python -m nnunet.inference.predict_simple \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    -t $TASK_ID \
    -tr nnUNetTrainerV2_fullGIL_mamba \
    -m 3d_fullres \
    -f $FOLD

# Step 4b: Resample predictions back to original spacing
echo ""
echo ">>> Resampling predictions to original Duying spacing..."
python "$(dirname "$0")/postprocess_predictions.py" \
    --pred_dir "$OUTPUT_DIR" \
    --duying_root "$DUYING_ROOT" \
    --output_dir "$FINAL_DIR"

echo ""
echo "============================================="
echo "Inference complete!"
echo "Predictions at original spacing: $FINAL_DIR"
echo "============================================="
