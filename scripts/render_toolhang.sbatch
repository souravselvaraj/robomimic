#!/bin/bash
#SBATCH -J fm_th_video
#SBATCH -A mmaghelih
#SBATCH -p short
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH -o /scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_th_video_%j.out

source /etc/profile.d/modules.sh 2>/dev/null
module load xvfb/focal || { echo "NO XVFB MODULE"; exit 1; }
which Xvfb xvfb-run
ldconfig -p | grep -c "libGL.so.1" || echo "NO libGL"

PYTHONPATH=/scratch/sselvaraj/projects/Flowmatch/robomimic \
xvfb-run -a -s "-screen 0 2560x2560x24" \
/scratch/sselvaraj/projects/Kuromoto-Diffusion-Policy/robomimic_base/.train_venv/bin/python -u \
    /scratch/sselvaraj/projects/Flowmatch/robomimic/scripts/render_video.py \
    --ckpt /scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_toolhang.pth \
    --dataset /scratch/sselvaraj/projects/Flowmatch/robomimic_data/tool_hang/ph/low_dim_v15.hdf5 \
    --horizon 700 \
    --video /scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_toolhang_rollout.mp4 \
    --video_skip 2
