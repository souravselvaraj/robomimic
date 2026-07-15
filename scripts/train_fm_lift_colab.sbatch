#!/bin/bash
#SBATCH -J fm_lift_colab
#SBATCH -A mmaghelih
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_lift_colab_%j.out

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd /scratch/sselvaraj/projects/Flowmatch/robomimic
VENV=/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/.train_venv

PYTHONPATH=$PWD "$VENV/bin/python" -u /scratch/sselvaraj/projects/Flowmatch/robomimic/scripts/colab_train_fm.py
