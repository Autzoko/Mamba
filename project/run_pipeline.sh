#!/bin/bash
# =============================================================================
# Master Pipeline: nnMamba cross-domain ABUS -> Duying
#
# This script orchestrates the entire pipeline end-to-end.
# Edit the paths below to match your HPC environment.
#
# Usage:
#   bash run_pipeline.sh [step]
#   Steps: 1 (convert), 2 (train), 3 (preprocess_duying), 4 (inference),
#          5 (evaluate), 6 (visualize), all (default)
# =============================================================================

set -e

# ==================== CONFIGURE THESE PATHS ====================

# Source data paths (read-only)
ABUS_ROOT="/scratch/$USER/data/ABUS/data"
DUYING_ROOT="/scratch/$USER/data/Duying"

# Working directories (will be created)
WORK_DIR="/scratch/$USER/nnmamba_pipeline"

# nnMamba directories
NNMAMBA_RAW="$WORK_DIR/nnMamba_raw/nnUNet_raw_data"
NNMAMBA_PREPROCESSED="$WORK_DIR/nnMamba_preprocessed"
NNMAMBA_RESULTS="$WORK_DIR/nnMamba_results"

# Intermediate outputs
DUYING_PREPROCESSED="$WORK_DIR/duying_preprocessed"
PREDICTIONS_ISOTROPIC="$WORK_DIR/predictions_isotropic"
PREDICTIONS_ORIGINAL="$WORK_DIR/predictions_original_spacing"
RESULTS_DIR="$WORK_DIR/results"
VIS_DIR="$WORK_DIR/visualization"

# Training parameters
TASK_ID=1
FOLD=0
GPU_ID=0

# ==================== END CONFIG ====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP="${1:-all}"

mkdir -p "$WORK_DIR"

run_step1() {
    echo "============ STEP 1: Data Conversion ============"
    python "$SCRIPT_DIR/data_conversion/convert_abus_to_nnmamba.py" \
        --abus_root "$ABUS_ROOT" \
        --output_base "$NNMAMBA_RAW" \
        --train_ratio 0.8 \
        --seed 42
}

run_step2() {
    echo "============ STEP 2: nnMamba Training ============"
    bash "$SCRIPT_DIR/training/train_nnmamba.sh" \
        --data_base "$WORK_DIR/nnMamba_raw" \
        --preprocessed "$NNMAMBA_PREPROCESSED" \
        --results "$NNMAMBA_RESULTS" \
        --task_id $TASK_ID \
        --fold $FOLD \
        --gpus $GPU_ID
}

run_step3() {
    echo "============ STEP 3: Preprocess Duying ============"
    python "$SCRIPT_DIR/inference/preprocess_duying.py" \
        --duying_root "$DUYING_ROOT" \
        --output_dir "$DUYING_PREPROCESSED" \
        --target_spacing 1.0 1.0 1.0 \
        --num_workers 8
}

run_step4() {
    echo "============ STEP 4: nnMamba Inference ============"
    bash "$SCRIPT_DIR/inference/run_inference.sh" \
        --input_dir "$DUYING_PREPROCESSED" \
        --output_dir "$PREDICTIONS_ISOTROPIC" \
        --final_dir "$PREDICTIONS_ORIGINAL" \
        --data_base "$WORK_DIR/nnMamba_raw" \
        --preprocessed "$NNMAMBA_PREPROCESSED" \
        --results "$NNMAMBA_RESULTS" \
        --duying_root "$DUYING_ROOT" \
        --task_id $TASK_ID \
        --fold $FOLD \
        --gpus $GPU_ID
}

run_step5() {
    echo "============ STEP 5: Evaluation ============"
    python "$SCRIPT_DIR/evaluation/evaluate_bbox.py" \
        --pred_dir "$PREDICTIONS_ORIGINAL" \
        --duying_root "$DUYING_ROOT" \
        --output_dir "$RESULTS_DIR"
}

run_step6() {
    echo "============ STEP 6: Visualization ============"
    python "$SCRIPT_DIR/visualization/visualize_failures.py" \
        --results_csv "$RESULTS_DIR/per_lesion_results.csv" \
        --duying_root "$DUYING_ROOT" \
        --pred_dir "$PREDICTIONS_ORIGINAL" \
        --output_dir "$VIS_DIR" \
        --n_worst 10
}

case "$STEP" in
    1) run_step1 ;;
    2) run_step2 ;;
    3) run_step3 ;;
    4) run_step4 ;;
    5) run_step5 ;;
    6) run_step6 ;;
    all)
        run_step1
        run_step2
        run_step3
        run_step4
        run_step5
        run_step6
        ;;
    *) echo "Unknown step: $STEP. Use 1-6 or 'all'"; exit 1 ;;
esac

echo ""
echo "Pipeline step '$STEP' complete!"
