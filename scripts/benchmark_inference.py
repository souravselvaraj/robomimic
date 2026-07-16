"""
Benchmark inference speed (one action-chunk generation) of flow matching vs.
diffusion policy on identical backbones. Weights are random unless checkpoints
are given - wall-clock time per chunk only depends on the number of network
evaluations, not on the weights.

Usage:
    PYTHONPATH=. python scripts/benchmark_inference.py --dataset <path> [--trials 50]

Compares:
    flow matching:    Euler 1 / 5 / 10 steps, midpoint 5 steps (10 evals)
    diffusion policy: DDPM 100 steps, DDIM 10 steps
"""
import argparse
import time

import numpy as np
import torch

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]


def build_model(algo_name, dataset_path, device, algo_overrides):
    config = config_factory(algo_name=algo_name)
    with config.unlocked():
        config.observation.modalities.obs.low_dim = list(OBS_KEYS)
        # the cosine LR scheduler needs the total step count (normally injected
        # by train.py); irrelevant for timing but required to build the model
        config.algo.optim_params["policy"]["num_train_batches"] = 100
        config.algo.optim_params["policy"]["num_epochs"] = 1
        for key, value in algo_overrides.items():
            node = config.algo
            *parents, leaf = key.split(".")
            for p in parents:
                node = node[p]
            node[leaf] = value
    ObsUtils.initialize_obs_utils_with_config(config)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config={"path": dataset_path},
        action_keys=["actions"],
        all_obs_keys=OBS_KEYS,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.set_eval()
    return model, shape_meta


def time_inference(model, shape_meta, device, trials):
    To = model.algo_config.horizon.observation_horizon
    obs_dict = {
        k: torch.randn((1, To) + tuple(shape_meta["all_shapes"][k]), device=device)
        for k in OBS_KEYS
    }
    times = []
    with torch.no_grad():
        for i in range(trials + 3):  # 3 warmup trials
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model._get_action_trajectory(obs_dict=obs_dict)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 3:
                times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/scratch/sselvaraj/projects/Flowmatch/robomimic_data/lift/ph/low_dim_v15.hdf5")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    variants = [
        ("flow matching, Euler 1 step", "flow_matching", {"fm.num_inference_steps": 1}),
        ("flow matching, Euler 5 steps", "flow_matching", {"fm.num_inference_steps": 5}),
        ("flow matching, Euler 10 steps", "flow_matching", {"fm.num_inference_steps": 10}),
        ("flow matching, midpoint 5 steps", "flow_matching", {"fm.num_inference_steps": 5, "fm.solver": "midpoint"}),
        ("diffusion policy, DDPM 100 steps", "diffusion_policy", {"ddpm.enabled": True, "ddim.enabled": False}),
        ("diffusion policy, DDIM 10 steps", "diffusion_policy", {"ddpm.enabled": False, "ddim.enabled": True}),
    ]

    print("device: {} | trials per variant: {} (after 3 warmup)\n".format(device, args.trials))
    results = []
    for name, algo_name, overrides in variants:
        model, shape_meta = build_model(algo_name, args.dataset, device, overrides)
        mean, std = time_inference(model, shape_meta, device, args.trials)
        results.append((name, mean, std))
        print("{:38s} {:8.2f} ms/chunk (+/- {:.2f})".format(name, mean * 1e3, std * 1e3))

    baseline = results[-2][1]  # DDPM 100
    print("\nspeedup vs DDPM-100:")
    for name, mean, _ in results:
        print("{:38s} {:6.1f}x".format(name, baseline / mean))
