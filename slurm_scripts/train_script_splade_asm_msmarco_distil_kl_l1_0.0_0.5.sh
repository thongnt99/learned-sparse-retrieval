#!/bin/bash
#SBATCH --nodes 1
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem 120G
#SBATCH -t 00-24:00:00
#SBATCH -o slurm_scripts/logs/splade_asm_msmarco_distil_kl_l1_0.0_0.5.out
#SBATCH -e slurm_scripts/logs/splade_asm_msmarco_distil_kl_l1_0.0_0.5.err
# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate lsr
echo "Training deepimpact using original settings"
export LSR_OUTPUT_PATH=""
export ANSERINI_OUTPUT_PAHT=""
python -m lsr.train +experiment=splade_asm_msmarco_distil_kl_l1_0.0_0.5 training_arguments.fp16=True wandb.resume=False