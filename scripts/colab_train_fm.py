"""
Train the flow matching policy on the lift (ph, low_dim) dataset, following the
structure of the official robomimic "Get Started" colab tutorial:
  1. download task dataset via DATASET_REGISTRY (done via scripts/download_datasets.py)
  2. create model with config_factory / algo_factory
  3. simple data loader with SequenceDataset
  4. simple training loop (epochs x gradient steps)
  5. evaluate the trained policy with run_rollout

Differences from the colab (which trains naive BC):
  - algo is "flow_matching" instead of "bc", so the data loader uses
    frame_stack/seq_length matching the flow matching horizons (2/16)
  - rollouts run without video since the cluster nodes have no GL libraries
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.utils.dataset import SequenceDataset
from robomimic.utils.train_utils import run_rollout

from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy


def get_standard_obs_keys(dataset_path):
    """
    Detect the standard low-dim observation keys present in the dataset:
    eef pose + gripper joints for every robot arm, plus object state.
    Handles both single-arm (robot0_*) and two-arm tasks like transport (robot1_*).
    """
    import h5py
    with h5py.File(dataset_path, "r") as f:
        demo0 = f["data"][list(f["data"].keys())[0]]
        keys = [
            k for k in demo0["obs"]
            if any(k.endswith(s) for s in ("eef_pos", "eef_quat", "gripper_qpos"))
            and "site" not in k
        ]
        keys.append("object")
    return sorted(keys)


def get_model(dataset_path, device, num_epochs, gradient_steps_per_epoch):
    """
    Use a default flow matching config to construct the model (colab section 3).
    """
    config = config_factory(algo_name="flow_matching")

    obs_keys = get_standard_obs_keys(dataset_path)

    # the cosine LR scheduler needs to know the total number of training steps
    # (normally injected by robomimic/scripts/train.py)
    with config.unlocked():
        config.algo.optim_params["policy"]["num_train_batches"] = gradient_steps_per_epoch
        config.algo.optim_params["policy"]["num_epochs"] = num_epochs
        config.observation.modalities.obs.low_dim = obs_keys

    # read config to set up metadata for observation modalities
    ObsUtils.initialize_obs_utils_with_config(config)

    # read dataset to get shape metadata for constructing the model
    # (this repo's v0.5 API takes a dataset config dict + action keys)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_config={"path": dataset_path},
        action_keys=["actions"],
        all_obs_keys=obs_keys,
    )

    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    return model, config


def get_data_loader(dataset_path, config, batch_size):
    """
    Get a data loader to sample batches of data (colab section 4).
    """
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=tuple(get_standard_obs_keys(dataset_path)),
        action_keys=["actions"],
        action_config={"actions": {"normalization": None}},
        dataset_keys=(
            "actions",
            "rewards",
            "dones",
        ),
        load_next_obs=False,
        frame_stack=config.algo.horizon.observation_horizon,
        seq_length=config.algo.horizon.prediction_horizon,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
    )
    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    return data_loader


def run_train_loop(model, data_loader, num_epochs, gradient_steps_per_epoch):
    """
    Simple training loop (colab section 4/5) - stripped-down version of
    robomimic's train.py without logging/checkpointing per epoch.
    """
    model.set_train()

    for epoch in range(1, num_epochs + 1):
        data_loader_iter = iter(data_loader)
        losses = []
        for _ in range(gradient_steps_per_epoch):
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)

            input_batch = model.process_batch_for_training(batch)
            info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=False)

            # step LR schedulers marked step_every_batch (robomimic's run_epoch
            # does this; without it the cosine warmup schedule stays at LR=0)
            for k, per_batch in model.step_lr_schedulers_every_batch.items():
                if per_batch and model.lr_schedulers[k] is not None:
                    model.lr_schedulers[k].step()

            step_log = model.log_info(info)
            losses.append(step_log["Loss"])

        model.on_epoch_end(epoch)
        lr = model.optimizers["policy"].param_groups[0]["lr"]
        print("Train Epoch {}: Loss {} LR {:.2e}".format(epoch, np.mean(losses), lr), flush=True)


def evaluate(model, config, dataset_path, num_rollouts, horizon):
    """
    Evaluate the trained policy in simulation (colab section 6, minus video -
    the cluster has no GL libraries for offscreen rendering).
    """
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    # stack observation frames to match the policy's observation horizon
    # (train.py does this via the same helper)
    env = EnvUtils.wrap_env_from_config(env, config=config)

    model.set_eval()
    policy = RolloutPolicy(model)

    successes = []
    for i in range(num_rollouts):
        rollout_log = run_rollout(
            policy=policy,
            env=env,
            horizon=horizon,
            render=False,
            video_writer=None,
            terminate_on_success=True,
        )
        successes.append(rollout_log["Success_Rate"])
        print("Rollout {}: {}".format(i + 1, rollout_log), flush=True)

    print("\n============= Evaluation =============")
    print("Success rate over {} rollouts: {}".format(num_rollouts, np.mean(successes)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/scratch/sselvaraj/projects/Flowmatch/robomimic_data/lift/ph/low_dim_v15.hdf5")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_rollouts", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--ckpt", default="/scratch/sselvaraj/projects/Flowmatch/robomimic/output/fm_lift_colab.pth")
    parser.add_argument("--eval_only", action="store_true", help="skip training and evaluate the saved checkpoint")
    args = parser.parse_args()

    assert os.path.exists(args.dataset), args.dataset
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    model, config = get_model(args.dataset, device, args.num_epochs, args.steps_per_epoch)

    if args.eval_only:
        model.deserialize(torch.load(args.ckpt, map_location=device, weights_only=False))
        print("loaded model from {}".format(args.ckpt))
    else:
        data_loader = get_data_loader(args.dataset, config, args.batch_size)
        run_train_loop(model, data_loader, num_epochs=args.num_epochs, gradient_steps_per_epoch=args.steps_per_epoch)

        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        torch.save(model.serialize(), args.ckpt)
        print("saved model to {}".format(args.ckpt))

    evaluate(model, config, args.dataset, num_rollouts=args.num_rollouts, horizon=args.horizon)
