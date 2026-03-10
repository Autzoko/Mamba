#!/bin/bash
#SBATCH --job-name=nnmamba_convert
#SBATCH --output=logs/step1_convert_%j.out
#SBATCH --error=logs/step1_convert_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

# Step 1: Convert ABUS data to nnMamba format (CPU only, I/O heavy)

module purge
module load python/3.10
source activate nnmamba  # adjust to your conda env name

cd /scratch/$USER/nnmamba_pipeline
bash /path/to/project/run_pipeline.sh 1
