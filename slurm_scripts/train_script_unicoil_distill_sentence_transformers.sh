#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu22
#SBATCH --gres gpu:a100:1
#SBATCH -t 00-12:00:00
#SBATCH -o slurm_scripts/logs/unicoil_distil.out
#SBATCH -e slurm_scripts/logs/unicoil_distil.err
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:

conda activate lsr
echo "Training Spade with MSMARCO triplets"
  python -m lsr.train +experiment=unicoil_msmarco_distil training_arguments.fp16=True training_arguments.per_device_train_batch_size=128  wandb.resume=False resume_from_checkpoint=False
