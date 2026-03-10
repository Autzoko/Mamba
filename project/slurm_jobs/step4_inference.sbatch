#!/bin/bash
#SBATCH --job-name=nnmamba_infer
#SBATCH --output=logs/step4_infer_%j.out
#SBATCH --error=logs/step4_infer_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Step 4: nnMamba inference on Duying (GPU required, depends on steps 2+3)

module purge
module load python/3.10
module load cuda/11.8
source activate nnmamba

cd /scratch/$USER/nnmamba_pipeline
bash /path/to/project/run_pipeline.sh 4
