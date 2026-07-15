# Conditional Flow Matching for robomimic

Design document for adding a Conditional Flow Matching (CFM) policy to this repo, alongside the existing Diffusion Policy implementation.

## Motivation

Diffusion Policy (already in this repo) learns a noise-prediction network and generates action sequences by iterative denoising (DDPM: 100 inference steps; DDIM: ~10). Conditional flow matching (Lipman et al. 2023, "Flow Matching for Generative Modeling"; Liu et al. 2022, "Rectified Flow") learns a **velocity field** along a simple probability path instead. Benefits:

- **Simpler training objective** — plain regression onto a closed-form target velocity; no noise schedules, no alpha/beta bookkeeping, no scheduler library needed for training.
- **Fast inference** — the learned ODE is integrated with a handful of Euler steps (5–10 typical); paths are near-straight so few steps suffice.
- **Drop-in compatibility** — same observation encoder, same `ConditionalUnet1D` backbone, same receding-horizon action queue as Diffusion Policy, so results are directly comparable.

## Method

Notation: `x1` = ground-truth normalized action sequence `[Tp, Da]`, `x0 ~ N(0, I)` noise, `t ~ U(0,1)` per-sample flow time, `obs_cond` = flattened encoded observation features over the observation horizon.

### Training (OT / rectified-flow path)

For each sample:

```
x_t = (1 - (1 - σ_min) · t) · x0 + t · x1        # linear interpolation path
u_t = x1 - (1 - σ_min) · x0                      # target (constant) velocity
loss = || v_θ(x_t, t, obs_cond) - u_t ||²
```

With `σ_min = 0` this is exactly rectified flow: straight lines from noise to data. `σ_min > 0` gives the OT-CFM path from Lipman et al. (small residual noise scale at t=1).

The velocity network `v_θ` is the same FiLM-conditioned `ConditionalUnet1D` used by Diffusion Policy — only the interpretation of the output changes (velocity instead of noise). Since the UNet's sinusoidal timestep embedding was designed for integer diffusion steps, `t ∈ [0,1]` is multiplied by a `time_embed_scale` (default 100) before embedding.

### Inference

Start from `x ~ N(0, I)` at `t=0` and integrate the ODE `dx/dt = v_θ(x, t, obs_cond)` to `t=1`:

- **Euler** (default): `x ← x + Δt · v_θ(x, t)` with `Δt = 1/N`, `N = num_inference_steps` (default 10).
- **Midpoint** (optional, 2 net evals/step): evaluate velocity at `x + ½Δt·v`, `t + ½Δt`.

Action selection then mirrors Diffusion Policy: predict `Tp` steps, execute `Ta` of them starting at index `To - 1`, refill the action queue when empty.

### Everything kept identical to Diffusion Policy

- Observation encoding: `ObservationGroupEncoder` + `TensorUtils.time_distributed`, BatchNorm→GroupNorm replacement (required for EMA).
- EMA of weights (`diffusers.training_utils.EMAModel`, power 0.75), used at inference.
- AdamW + cosine LR schedule with warmup.
- Actions must be normalized to `[-1, 1]` (`hdf5_normalize_action`); checked on first batch.
- Horizons: `To=2`, `Ta=8`, `Tp=16`; `train.seq_length=16`, `train.frame_stack=2`.

## Files

| File | Change |
|---|---|
| `robomimic/algo/flow_matching.py` | **New.** `FlowMatchingUNet(PolicyAlgo)` + `@register_algo_factory_func("flow_matching")`. Reuses `replace_bn_with_gn` from `diffusion_policy.py`. |
| `robomimic/config/flow_matching_config.py` | **New.** `FlowMatchingConfig(BaseConfig)`, `ALGO_NAME = "flow_matching"`. |
| `robomimic/algo/__init__.py` | Import `FlowMatchingUNet` to register it. |
| `robomimic/config/__init__.py` | Import `FlowMatchingConfig` to register it. |
| `robomimic/exps/templates/flow_matching.json` | **Generated** via `scripts/generate_config_templates.py`. |

## Config (new `algo.fm` section)

```
algo.fm.sigma_min = 0.0            # 0 = rectified flow; >0 = OT-CFM path
algo.fm.num_inference_steps = 10   # ODE integration steps
algo.fm.solver = "euler"           # "euler" | "midpoint"
algo.fm.time_embed_scale = 100.0   # scales t∈[0,1] before sinusoidal embedding
```

The `unet`, `ema`, `horizon`, and `optim_params` sections match `diffusion_policy_config.py` exactly (UNet kwargs are additionally passed through: `diffusion_step_embed_dim`, `down_dims`, `kernel_size`, `n_groups`).

## Usage

```bash
python robomimic/scripts/train.py \
    --config robomimic/exps/templates/flow_matching.json \
    --dataset /path/to/demo.hdf5
```

(Enable `train.hdf5_normalize_action` as with Diffusion Policy.)

## Status

The implementation described above is **already in place** and passed a synthetic smoke test (loss computation and backprop, EMA update, Euler and midpoint inference, action-queue refill, serialize/deserialize roundtrip) using the venv at `Kuromoto-Diffusion-Policy/robomimic_base/.train_venv` (torch 2.13, diffusers 0.11.1). Not yet validated: an actual training run on a real dataset.

## Possible extensions (not implemented)

- Non-uniform `t` sampling (e.g. Beta / logit-normal emphasis near t=0) — often helps in practice (used by π0, SD3).
- Transformer backbone variant (`algo.transformer`), matching the Diffusion Policy TODO.
- Higher-order solvers (Heun/RK4) or adaptive step ODE solvers.
- Distillation to 1-step (ReFlow / consistency-style) for real-time control.
