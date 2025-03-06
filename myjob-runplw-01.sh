#!/bin/bash

#SBATCH --job-name=myjob-runplw-01
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=30G
#SBATCH --mail-user=priscilla.ruth.chyrva@students.uni-mannheim.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/ceph/pratz/GitHub_repos/master-thesis/external/prompt-loss-weight
#SBATCH --partition=gpu-vram-48gb

# Load necessary modules (if any)
module load cuda/12.1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate plw_env

# Run your Python script
python run_plw.py \
    --model_id codellama/CodeLlama-7b-Instruct-hf \
    --dataset_id ruthchy/semantic-length-generalization-logo-data-desc-ascii_35 \
    --prompt_template "[INST] Here is a gray scale image represented with integer values 0-9:\n{ASCII-Art}\nPlease write a Python program that generates this image using our custom turtle module.[/INST]" \
    --completion_template "{Program}" \
    --prompt_loss_weight 0.1 \
    --data_dir ~/data \
    --rand_seed 42 \
    --max_seq_length 2048 \
    --learning_rate 2e-4 \
    --subsample_train 8006.0 \
    --subsample_eval 997.0 \
    --max_samples 500000 \
    --save_steps 500 \
    --eval_steps 100 \
    --logging_steps 5 \
    --lora_alpha 64 \
    --lora_r 64 \
    --lora_dropout 0.0 \
    --push_to_hub True \
    --hub_model_id f"CodeLlama-7b-Inst-Length-Gen-Logo-Ascii_35"\
    --resume_from_checkpoint True