#!/bin/bash
#SBATCH -J fm_square
#SBATCH -A mmaghelih
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -o /scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_square_%j.out

export WANDB_CONSOLE=off
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd /scratch/sselvaraj/projects/Flowmatch/robomimic
VENV=/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/.train_venv

# resume if a previous run of this experiment left a checkpoint; otherwise
# answer train.py's "overwrite existing experiment dir?" prompt for fresh runs
EXP_DIR=/scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_square_lowdim
if [[ -n "$(find "$EXP_DIR" -maxdepth 2 -name last.pth 2>/dev/null)" ]]; then
  echo "Found existing checkpoint under $EXP_DIR — resuming."
  PYTHONPATH=$PWD "$VENV/bin/python" -u robomimic/scripts/train.py \
      --config /scratch/sselvaraj/projects/Flowmatch/robomimic/configs/fm_square_lowdim.json --resume
else
  printf 'y\n' | PYTHONPATH=$PWD "$VENV/bin/python" -u robomimic/scripts/train.py \
      --config /scratch/sselvaraj/projects/Flowmatch/robomimic/configs/fm_square_lowdim.json
fi
