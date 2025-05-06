#!/bin/bash

#SBATCH --job-name=hp-job-pipeline
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --mail-user=priscilla.ruth.chyrva@students.uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --chdir=/ceph/pratz/GitHub_repos/master-thesis
#SBATCH --partition=gpu-vram-48gb

# Load necessary modules
module load cuda/12.8

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate thesis_env

# Check if we're already done
if [ -f "hp_tuning_completed.marker" ]; then
    echo "Hyperparameter tuning already completed according to marker file."
    echo "To re-run tuning, delete the hp_tuning_completed.marker file."
    exit 0
fi

# Set the number of trials and timeout for Optuna
N_TRIALS=24
TIMEOUT=$((48 * 60 * 60)) # Calculate timeout in seconds (48 hours) can also be set to None (no timeout, just n_trails will limit duration

# Run hyperparameter tuning script
python hp_optuna.py --n_trials $N_TRIALS #--timeout $TIMEOUT --test_mode # if this flag is set each configuration will run for a single epoch & use only a fraction of the data

# Chain the next job if needed (unless we're done)