#!/bin/bash
# Train one benchmark config: sbatch scripts/train_benchmark.sbatch configs/benchmark/<name>.json
# Resumes from last.pth automatically if the job was requeued/killed and rerun.
#SBATCH -J fm_bench
#SBATCH -A mmaghelih
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/sselvaraj/projects/Flowmatch/robomimic/output/benchmark_%x_%j.out

set -u
export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd /scratch/sselvaraj/projects/Flowmatch/robomimic
VENV=/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/.train_venv

CONFIG="$1"
NAME=$(basename "$CONFIG" .json)
EXP_DIR=/scratch/sselvaraj/projects/Flowmatch/robomimic/output/benchmark/"$NAME"

if [[ -n "$(find "$EXP_DIR" -maxdepth 2 -name last.pth 2>/dev/null)" ]]; then
  echo "Found existing checkpoint under $EXP_DIR — resuming."
  PYTHONPATH=$PWD "$VENV/bin/python" -u robomimic/scripts/train.py --config "$CONFIG" --resume
else
  printf 'y\n' | PYTHONPATH=$PWD "$VENV/bin/python" -u robomimic/scripts/train.py --config "$CONFIG"
fi
