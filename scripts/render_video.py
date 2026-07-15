"""
Render rollout videos of the trained flow matching policy (colab section 6).

Run with: MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python render_video.py
"""
import os
import sys
import argparse
import numpy as np
import torch
import imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Compute nodes have no libEGL, but robosuite force-sets MUJOCO_GL=egl at import
# unless it is exactly "osmesa" or "glx". Preferred path on this cluster: run with
#   MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
#   LD_LIBRARY_PATH=<spack mesa lib dir with libOSMesa.so.8>
# (the mesa/25.0.5 module does NOT set LD_LIBRARY_PATH itself). Fallback for
# EGL-capable hosts: import mujoco under "glfw", then switch to "glx" so
# robosuite's override leaves it alone and uses its GLFW context under Xvfb.
if os.environ.get("MUJOCO_GL") != "osmesa":
    os.environ["MUJOCO_GL"] = "glfw"
    import mujoco  # noqa: F401
    os.environ["MUJOCO_GL"] = "glx"

from colab_train_fm import get_model

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.train_utils import run_rollout
from robomimic.algo import RolloutPolicy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/scratch/sselvaraj/projects/Flowmatch/robomimic_data/lift/ph/low_dim_v15.hdf5")
    parser.add_argument("--ckpt", default="/scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_lift_colab_v2.pth")
    parser.add_argument("--video", default="/scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_lift_rollout.mp4")
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--video_skip", type=int, default=5)
    args = parser.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    model, config = get_model(args.dataset, device, 1000, 100)
    model.deserialize(torch.load(args.ckpt, map_location=device, weights_only=False))
    print("loaded model from {}".format(args.ckpt))

    env_meta = FileUtils.get_env_metadata_from_dataset(args.dataset)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,   # needed for video frames
        use_image_obs=False,
    )
    env = EnvUtils.wrap_env_from_config(env, config=config)  # frame stacking

    model.set_eval()
    policy = RolloutPolicy(model)

    video_writer = imageio.get_writer(args.video, fps=20)
    successes = []
    for i in range(args.num_rollouts):
        rollout_log = run_rollout(
            policy=policy,
            env=env,
            horizon=args.horizon,
            render=False,
            video_writer=video_writer,
            video_skip=args.video_skip,
            terminate_on_success=True,
        )
        successes.append(rollout_log["Success_Rate"])
        print("Rollout {}: {}".format(i + 1, rollout_log), flush=True)
    video_writer.close()

    print("success rate: {}".format(np.mean(successes)))
    print("saved video to {}".format(args.video))
