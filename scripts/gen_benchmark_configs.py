"""
Generate training configs for the flow matching vs. diffusion policy benchmark:
{flow_matching, diffusion_policy} x {lift, can, square, tool_hang, transport} x 3 seeds,
all on ph low_dim datasets at a matched training budget (1000 epochs x 100 steps,
batch 256, rollout eval every 100 epochs).

Configs are written to configs/benchmark/. Submit them with
scripts/submit_benchmark.sh (or sbatch scripts/train_benchmark.sbatch <config>).
"""
import json
import os

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.abspath(os.path.join(REPO_DIR, "..", "robomimic_data"))
OUT_DIR = os.path.join(REPO_DIR, "configs", "benchmark")

ALGOS = {
    "fm": "flow_matching",
    "dp": "diffusion_policy",
}

# task -> (rollout horizon, num robot arms); horizons follow the robomimic benchmark
TASKS = {
    "lift": (400, 1),
    "can": (400, 1),
    "square": (400, 1),
    "tool_hang": (700, 1),
    "transport": (700, 2),
}

SEEDS = [1, 2, 3]


def obs_keys(num_arms):
    keys = []
    for i in range(num_arms):
        keys += [
            "robot{}_eef_pos".format(i),
            "robot{}_eef_quat".format(i),
            "robot{}_gripper_qpos".format(i),
        ]
    return keys + ["object"]


def make_config(algo_short, task, seed):
    horizon, num_arms = TASKS[task]
    name = "{}_{}_seed{}".format(algo_short, task, seed)
    return name, {
        "algo_name": ALGOS[algo_short],
        "experiment": {
            "name": name,
            "validate": True,
            "logging": {
                "terminal_output_to_txt": True,
                "log_tb": True,
                "log_wandb": False,
            },
            "save": {
                "enabled": True,
                "every_n_epochs": 100,
                "on_best_rollout_success_rate": True,
            },
            "epoch_every_n_steps": 100,
            "rollout": {
                "enabled": True,
                "n": 20,
                "horizon": horizon,
                "rate": 100,
            },
            "render_video": False,
        },
        "train": {
            "data": os.path.join(DATA_DIR, task, "ph", "low_dim_v15.hdf5"),
            "output_dir": os.path.join(REPO_DIR, "output", "benchmark"),
            "hdf5_filter_key": "train",
            "hdf5_validation_filter_key": "valid",
            "num_epochs": 1000,
            "batch_size": 256,
            "seed": seed,
        },
        "observation": {
            "modalities": {
                "obs": {
                    "low_dim": obs_keys(num_arms),
                    "rgb": [],
                }
            }
        },
    }


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for algo_short in ALGOS:
        for task in TASKS:
            dataset = os.path.join(DATA_DIR, task, "ph", "low_dim_v15.hdf5")
            assert os.path.exists(dataset), "missing dataset: {}".format(dataset)
            for seed in SEEDS:
                name, config = make_config(algo_short, task, seed)
                path = os.path.join(OUT_DIR, name + ".json")
                with open(path, "w") as f:
                    json.dump(config, f, indent=4)
                print("wrote {}".format(os.path.relpath(path, REPO_DIR)))
