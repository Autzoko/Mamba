#!/bin/bash
#SBATCH --job-name=nnmamba_train
#SBATCH --output=logs/step2_train_%j.out
#SBATCH --error=logs/step2_train_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Step 2: nnMamba plan+preprocess+train (GPU required)

module purge
module load python/3.10
module load cuda/11.8
source activate nnmamba

cd /scratch/$USER/nnmamba_pipeline
bash /path/to/project/run_pipeline.sh 2
