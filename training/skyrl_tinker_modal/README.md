# Self-Hosting the Training Backend with SkyRL and Modal

The training package supports SkyRL's Tinker server on Modal GPUs. Each launcher requests GPUs and allocates them between the learner and sampler for the corresponding experiment. [SkyRL's Tinker server](https://docs.skyrl.ai/docs/tinker/overview)

| Launch script | Use it for | GPUs | Layout |
|---|---|---|---|
| `serve_number_search_sync.py` | Synchronous Number Search | 2x H100 | colocated: both GPUs train *and* sample |
| `serve_number_search_async.py` | Asynchronous Number Search | 2x H100 | split: 1 trains, 1 samples |
| `serve_textcraft.py` | TextCraft training and evaluation | 4x H100 | split: 1 trains, 3 sample |

Synchronous training alternates between sampling and updating, so both GPUs can handle both phases. Asynchronous training samples during policy updates and uses separate GPUs for the learner and sampler. TextCraft episodes spend most of their time sampling, so its launcher allocates three of the four GPUs to inference.

The shared parts of the three deployments (the container image, the SkyRL version, the optimizer settings) live in `skyrl_server.py`. All three train with SkyRL's Megatron backend, which is the one SkyRL supports Qwen3.5 on, and sample with vLLM. The image pins a revision of [our SkyRL fork](https://github.com/adityasoni9998/SkyRL) containing the Megatron token-batched output-ordering fix and adapter-only sampler checkpoints, so that a saved LoRA snapshot is your adapter rather than a copy of the whole model. The image also applies `patches/result_persistence.patch` to handle failed sampling-result database writes, as described below.

## Sampling requests stuck after database failures

The pinned server can leave a sampling request pending forever if generation
finishes but saving its result fails. The October 26 server log shows SQLite
`database is locked`, exhausted SQLAlchemy connections, and unhandled
`call_and_store_result` exceptions. Clients then wait until their 7,200-second
no-progress timeout, even after the database starts accepting new requests.

The local patch serializes sampling completion writes within the API process,
serializes result JSON before opening the database transaction, and retries
SQLite lock and connection-pool errors using fresh sessions. It allows up to
six attempts within a 180-second persistence budget (database cancellation
cleanup can take additional time). Generation itself is not repeated. If
persistence still fails, existing and subsequent result polls receive the
server's normal terminal error response without relying on a working database.
This fallback is held in API-process memory and does not survive a restart.
Background forwarding tasks are also tracked and awaited on shutdown.

This addresses stranded result requests; it does not establish the cause of
the original database lock or fix the separate truncated HTTP responses in
the logs. It does not change checkpoint storage or the training algorithm.
An ongoing database failure can still fail training promptly.

The patch is applied after the cached CUDA dependency build and checked against
the pinned source. It takes effect only after redeployment. With no training
run active, deploy TextCraft from this directory:

```bash
uv run --locked --extra modal modal deploy serve_textcraft.py
```

CPU-only fault-injection tests can run against a checkout of the pinned SkyRL
commit with the patch applied. From the training directory:

```bash
git -C /path/to/SkyRL apply "$PWD/skyrl_tinker_modal/patches/result_persistence.patch"
SKYRL_SOURCE_ROOT=/path/to/SkyRL uv run --no-project \
  --with 'sqlalchemy[asyncio]' --with sqlmodel --with aiosqlite --with pytest \
  python -m pytest skyrl_tinker_modal/tests/test_result_persistence.py -q
```

The tests load the patched persistence and retrieval methods without importing
the GPU stack. They exercise real SQLite write locks, concurrent completions,
pool exhaustion, terminal error delivery, missing rows, and cancellation.

## One-Time Setup

Create a Modal account and select a workspace with enough H100 capacity for your launcher. Run these commands from `training/`.

```bash
uv sync --locked --python 3.12 --extra modal
uv run --locked --extra modal modal setup
uv run --locked --extra modal modal profile current
uv run --locked --extra modal modal token info
```

Include `--extra modal` on commands that need the Modal CLI. The trainer itself only needs the Tinker client.

The first deployment builds a container image with SkyRL, Megatron, and their CUDA dependencies on an H100. The launcher profiles share the cached image. The optional installation check also uses one H100.

```bash
(cd skyrl_tinker_modal && uv run --locked --extra modal modal run serve_number_search_sync.py::smoke_install)
```

## Deploy and connect

**1. Deploy the launcher that matches your YAML.** From `training/`, deploy the synchronous Number Search server like this.

```bash
(cd skyrl_tinker_modal && uv run --locked --extra modal modal deploy serve_number_search_sync.py)
```

Use the asynchronous Number Search launcher for `configs/number_search_async*.yaml` and the TextCraft launcher for `configs/textcraft_grpo.yaml`. The server defaults to `Qwen/Qwen3.5-4B`. If you set `SKYRL_BASE_MODEL` before deployment, use that exact model ID in the training config as well.

The included launchers expose a public Modal endpoint. Anyone with the URL can send requests while the app is deployed. A workspace that blocks public web functions will reject this deployment. [Modal web endpoint authentication](https://modal.com/docs/guide/webhook-proxy-auth)

**2. Set the client endpoint.** Still in `training/`, replace the example with the exact `tinker_server` URL printed by the deployment. Use the base URL without a `/v1` suffix. Unset a hosted Tinker key when using this SkyRL deployment.

```bash
export TINKER_BASE_URL='https://YOUR-DEPLOYMENT.modal.run'
unset TINKER_API_KEY
```

The first request starts the container and loads the model. Once `TINKER_BASE_URL` is set, run the connection check and the short probe in the training guide. [Training guide](../README.md#connect-to-a-backend)

**3. Stop the server when you finish.** From `training/`, list the apps and stop the one you deployed.

```bash
uv run --locked --extra modal modal app list
uv run --locked --extra modal modal app stop agent-training-ns-sync
```

Stopping removes the deployment, so run `uv run modal deploy ...` again before your next run. This takes seconds because the image is cached, and the URL stays the same. Your checkpoints live on a volume and are not affected.

The app names are `agent-training-ns-sync`, `agent-training-ns-async`, and `agent-training-textcraft`. You can see what is running, and what it has cost so far, in the Modal dashboard.

## Run profiles in parallel

The three launch scripts are separate Modal apps, each with its own URL, GPUs, and checkpoint volume, so you can run one part on each of them at the same time without any extra setup. You only need a second app name when you want two runs on the *same* launcher, for example `p2a` and `p41`, which both use the synchronous server. Set `SKYRL_APP_NAME` when deploying either Number Search launcher:

```bash
SKYRL_APP_NAME=agent-training-reinforce uv run modal deploy serve_number_search_sync.py
SKYRL_APP_NAME=agent-training-grpo-g2 uv run modal deploy serve_number_search_sync.py
```

Each app has its own endpoint and checkpoint/database volume. In separate terminals, set `TINKER_BASE_URL` to the corresponding deployment URL and launch one training run per app. Without `SKYRL_APP_NAME`, the default app names are unchanged. Use the same app name when resuming a run, and stop each app with `uv run modal app stop <app name>` when you are done.

## Trainer and Inference Settings

Each `serve_*.py` file combines `BASE_BACKEND_CONFIG` from [skyrl_server.py](skyrl_server.py) with its profile overrides. The settings below are documented for reference; you should not need to change them for any of the required runs. A server's settings live in these files, not in the client's training YAML, and a change only takes effect when you redeploy (a change to `BASE_BACKEND_CONFIG` affects all three servers).

These are the SkyRL backend configuration flags:

| Flag | Meaning: value used here |
|---|---|
| `trainer.placement.policy_num_nodes` | no. of GPU nodes to use: `1` |
| `trainer.policy.megatron_config.{tensor,pipeline,context}_model_parallel_size` | Megatron model, pipeline and context parallelism: `1` (a 4B model with LoRA fits on one GPU, so we only use data parallelism) |
| `trainer.placement.policy_num_gpus_per_node` | no. of GPUs to allocate to the Megatron training backend: `2` for synchronous RL + number search, `1` for the others. |
| `trainer.placement.colocate_all` | share the same GPUs between trainer and inference engines: `True` for synchronous RL where training and sampling alternate, `False` for async RL. |
| `generator.inference_engine.num_engines` | no. of vLLM inference engines to host: `2` for synchronous RL + number search, `1` for async RL + number search, `3` for TextCraft. Equal to the no. of inference GPUs as we use data parallelism. |
| `trainer.policy.language_model_only` | train only the text backbone as Qwen3.5 is a multimodal LLM: `True` (our tasks are text-only) |
| `generator.inference_engine.language_model_only` | load only the text backbone for vLLM: `True` (our tasks are text-only) |
| `trainer.policy.megatron_config.lora_config.merge_lora` | merge the LoRA adapter into the base model: `False` (keep it separate for sampling and adapter checkpoints) |
| `trainer.policy.optimizer_config.max_grad_norm` | clip the gradient norms to this value: `1.0`. |
| `trainer.policy.optimizer_config.adam_betas` | decay rates for Adam's first- and second-moment estimates: `[0.9, 0.95]` |
| `trainer.policy.optimizer_config.weight_decay` | weight decay in Adam optimizer: `0.0` (disabled) |
| `trainer.use_expandable_segments` | PyTorch's expandable-segments allocator on the training workers, which reduces memory fragmentation: `True` |
| `trainer.max_tokens_per_microbatch` | max. number of tokens per training micro-batch for [dynamic batching](https://docs.nvidia.com/nemo/rl/latest/design-docs/sequence-packing-and-dynamic-batching.html): `32768` (SkyRL bin-packs a variable number of whole sequences into each micro-batch by token count instead of using a fixed number of sequences; this is a soft cap, a single sequence longer than this runs alone in its own micro-batch) |
| `trainer.fused_lm_head_logprob` | compute logprobs in chunks inside the output layer without storing the full-vocabulary logits: `True` for all parts (the unfused backward pass runs out of GPU memory at this micro-batch size) |
| `trainer.logprobs_chunk_size` | no. of tokens per logprob chunk: `1024`, explicitly set for the Number Search servers and the SkyRL default for TextCraft. |
| `generator.inference_engine.gpu_memory_utilization` | fraction of each inference GPU's memory that vLLM may use for weights, KV cache and activations: `0.7` for colocated setups (leave room for the trainer), `0.85` otherwise. |
| `generator.inference_engine.engine_init_kwargs.max_model_len` | context window in vLLM (max allowed prompt tokens + output tokens): `32768` for number search, `65000` for TextCraft. A request over this limit is rejected; the framework then abandons that episode with reward 0 and the run carries on (`agent_training/rollout.py`). |
| `generator.inference_engine.max_num_batched_tokens` | max. allowed tokens in a single inference batch for each vLLM engine: `50000` (50K) for all server configs (independent of the training microbatch budget) |
| `generator.inference_engine.max_num_seqs` | max. allowed sequences in a single inference batch for each vLLM engine: `256` |
| `generator.inference_engine.enable_prefix_caching` | enable KV-caching in vLLM for faster inference: `True` |
| `generator.inference_engine.use_expandable_segments` | the same allocator setting for the vLLM processes: `False` |
| `generator.inference_engine.enforce_eager` | disable [CUDA graphs](https://github.com/vllm-project/vllm/issues/4449) and use eager PyTorch execution for inference (can reduce memory usage but is slower): `False` |
| `generator.inference_engine.engine_init_kwargs.gdn_prefill_backend` | kernel backend to use for Qwen3.5's Gated-DeltaNet prefill: `"triton"` |
| `generator.inference_engine.engine_init_kwargs.compilation_config` | CUDA graph configuration: `{"cudagraph_mode": "FULL_DECODE_ONLY"}` (decoding only) |
| `generator.inference_engine.enable_ray_prometheus_stats` | expose inference statistics through Ray's Prometheus integration: `True` |
| `generator.inference_engine.engine_init_kwargs.enable_mfu_metrics` | log model FLOPs utilization during inference: `True` |

Tinker requests supply the learning rate; the optimizer flags above are configured on the server.

Settings for the three parts:

| Profile | GPU placement | vLLM memory fraction | Context limit | Fused logprobs | Rationale |
|---|---|---|---|---|---|
| Number Search sync | 2 H100s shared | `0.7` | `32768` | `True` | Training and sampling alternate; leave memory for both. |
| Number Search async | 1 training + 1 inference H100 | `0.85` | `32768` | `True` | Separate GPUs for concurrent training and sampling. |
| TextCraft | 1 training + 3 inference H100s | `0.85` | `65000` | `True` | Longer episodes need more context and sampling capacity. |

If you ever change GPU placement, keep it consistent with Modal's GPU allocation. All three servers also request 8 CPUs and 32 GiB RAM, with a 12-hour server timeout and a five-minute idle scaledown window.

## What It Costs

Image builds and live GPU containers can incur charges. The two-GPU Number Search and four-GPU TextCraft servers scale to zero after five idle minutes. [Modal pricing](https://modal.com/pricing)

- Get your code passing the tests locally before you deploy anything. The tests need no GPU.
- Do a short probe before every full run. A crash at step 2 of a probe is much cheaper than a crash at step 40 of the real thing.
- Stop the app when you step away, and never leave the TextCraft server up overnight.
