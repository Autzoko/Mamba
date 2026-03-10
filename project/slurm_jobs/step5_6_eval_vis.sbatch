#!/bin/bash
#SBATCH --job-name=nnmamba_eval
#SBATCH --output=logs/step5_6_eval_%j.out
#SBATCH --error=logs/step5_6_eval_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

# Steps 5+6: Evaluation and visualization (CPU only, depends on step 4)

module purge
module load python/3.10
source activate nnmamba

cd /scratch/$USER/nnmamba_pipeline
bash /path/to/project/run_pipeline.sh 5
bash /path/to/project/run_pipeline.sh 6
