# Agent training

`agent_training` fine-tunes tool-using policies with REINFORCE and Dr.GRPO. The synchronous trainer collects batches before each update. The asynchronous trainer streams rollout groups through a bounded queue while the learner updates, tracks policy versions, and drops groups older than the configured staleness limit. The checked-in configurations include an importance-weighting ablation.

Run commands from this directory. The offline test suite exercises reward handling, rollout batching, and trainer behavior with mock policies.

```bash
uv sync --locked
uv run --locked pytest
```

Live runs need a Tinker-compatible service. Set `TINKER_API_KEY` and, for a self-hosted service, `TINKER_BASE_URL`. See the [SkyRL deployment guide](skyrl_tinker_modal/README.md) for the Modal setup.

```bash
uv run --locked python scripts/train_sync.py --config configs/number_search_grpo.yaml
uv run --locked python scripts/train_async.py --config configs/textcraft_grpo.yaml
uv run --locked python scripts/evaluate_checkpoint.py --config configs/textcraft_grpo.yaml \
  --model-path tinker://YOUR_CHECKPOINT --k 4
```

Training writes metrics, sampled trajectories, and checkpoint references under `runs/`. The evaluation script supports the base model, a Tinker checkpoint, or an exported LoRA adapter served by vLLM or SGLang. See [dataset splits](data/README.md) and [run configurations](configs).
