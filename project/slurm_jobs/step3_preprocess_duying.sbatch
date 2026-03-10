#!/bin/bash
#SBATCH --job-name=nnmamba_preproc_duying
#SBATCH --output=logs/step3_preproc_%j.out
#SBATCH --error=logs/step3_preproc_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

# Step 3: Preprocess Duying volumes (CPU only, can run in parallel with step 2)

module purge
module load python/3.10
source activate nnmamba

cd /scratch/$USER/nnmamba_pipeline
bash /path/to/project/run_pipeline.sh 3
