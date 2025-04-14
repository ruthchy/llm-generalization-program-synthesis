#!/bin/bash

#SBATCH --job-name=job-ft-pipeline.sh
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=30G
#SBATCH --mail-user=priscilla.ruth.chyrva@students.uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --chdir=/ceph/pratz/GitHub_repos/master-thesis
#SBATCH --partition=gpu-vram-48gb

# Load necessary modules (if any)
module load cuda/12.8

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_env

# Run your Python script
python pipeline.py \
    --fine_tune \
    --sample_fraction 1.0 \
    --config "config.yaml" \
    --wb_type "test"