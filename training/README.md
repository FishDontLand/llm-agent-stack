# Agent training

`agent_training` fine-tunes tool-using policies with REINFORCE and Dr.GRPO. The synchronous trainer collects batches before each update. The asynchronous trainer streams rollout groups through a bounded queue while the learner updates, tracks policy versions, and drops groups older than the configured staleness limit. The checked-in configurations include an importance-weighting ablation.

## Install and test

Run commands from `training/`. The offline suite exercises reward handling, rollout batching, and trainer behavior with mock policies.

```bash
uv sync --locked --python 3.12
uv run --locked pytest
```

## Connect to a backend

Choose a backend before running the probe or a full experiment. The checked-in YAMLs use `Qwen/Qwen3.5-4B`.

- **SkyRL on Modal.** Install the `modal` extra, sign in, and deploy the launcher that matches your config. Export the exact URL printed by the deployment as `TINKER_BASE_URL`. The included SkyRL service uses a placeholder Tinker key automatically when `TINKER_API_KEY` is unset. [Deployment commands and GPU requirements](skyrl_tinker_modal/README.md#one-time-setup)
- **Hosted Tinker.** Set `TINKER_API_KEY` to your hosted service key and unset `TINKER_BASE_URL`. Check that your service offers the base model required by the YAML. [Hosted Tinker authentication](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)

Check the connection and the models the server reports before starting an experiment. This starts a cold Modal server if you use SkyRL.

```bash
uv run --locked python - <<'PY'
from agent_training.tinker_adapter import make_service_client

capabilities = make_service_client().get_server_capabilities()
for model in capabilities.supported_models:
    print(model.model_name)
PY
```

## Run a short Number Search probe

Use the synchronous Number Search backend for this two-step REINFORCE run. It checks sampling and training with two tasks per step, and skips the baseline and periodic validation passes.

```bash
uv run --locked python scripts/train_sync.py --config configs/number_search_reinforce.yaml \
  --set train.run_name=number_search_probe --set train.num_steps=2 \
  --set train.tasks_per_batch=2 --set train.eval_every=0 \
  --set train.trajectory_log_every=1
```

Look for training records in `runs/number_search_probe.jsonl`, rollout transcripts under `runs/number_search_probe_trajectories/`, and `final_sampler_path` in `runs/number_search_probe.checkpoints.json`. A log record with `train/skipped` indicates that step had no usable training sequences. Choose a new run name when repeating a probe.

## Train and evaluate

For SkyRL on Modal, switch `TINKER_BASE_URL` to the matching deployment before changing profiles.

```bash
# Synchronous Number Search server
uv run --locked python scripts/train_sync.py --config configs/number_search_grpo.yaml

# Asynchronous Number Search server
uv run --locked python scripts/train_async.py --config configs/number_search_async.yaml

# TextCraft server
uv run --locked python scripts/train_async.py --config configs/textcraft_grpo.yaml
```

For checkpoint evaluation, use the same backend deployment and take the sampler URI from the completed run's checkpoint manifest. This example evaluates five held-out TextCraft tasks with one rollout per task.

```bash
checkpoint='tinker://REPLACE_WITH_FINAL_SAMPLER_PATH'
uv run --locked python scripts/evaluate_checkpoint.py --config configs/textcraft_grpo.yaml \
  --model-path "$checkpoint" --num-tasks 5 --k 1 --concurrency 2
```

The evaluation script prints JSON with success rate and pass@k. Drop the task limit for the full validation split or select a different split with `--tasks`. To resume a run with a saved optimizer checkpoint, keep the run name and backend and add `--set train.resume=true` to the original training command. For optional Weights & Biases logging, set `WANDB_API_KEY` and pass `--set train.wandb_project=YOUR_PROJECT`. [Dataset splits](data/README.md) · [Run configurations](configs)
