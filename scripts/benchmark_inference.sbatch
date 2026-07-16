#!/bin/bash
#SBATCH -J fm_bench_infer
#SBATCH -A mmaghelih
#SBATCH -p short,long,quick
#SBATCH --gres=gpu:1
#SBATCH --constraint="RTX6000B|A100|H100"
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0:30:00
#SBATCH -o /scratch/sselvaraj/projects/Flowmatch/robomimic/output/bench_infer_%j.out

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd /scratch/sselvaraj/projects/Flowmatch/robomimic
VENV=/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/.train_venv

nvidia-smi --query-gpu=name --format=csv,noheader
PYTHONPATH=$PWD "$VENV/bin/python" -u scripts/benchmark_inference.py --trials 100
