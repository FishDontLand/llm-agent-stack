# LLM Agent Stack

I built three pieces of the LLM agent development workflow: a ReAct harness for running tool-using agents, an LLM-as-judge evaluator for chart-generation runs, and a reinforcement learning trainer that improves tool use from environment rewards. Each component is an importable Python package with its own dependencies and runnable entry points.

| Component | What I implemented | Where to start |
| --- | --- | --- |
| **Agent harness** | ReAct execution loop, tool dispatch, skill loading, trajectory logging, and context compaction. Ran the harness on software repair and chess, with a Modal sandbox for tool execution. | [Code](harness/src/agent_harness/agent), [run guide](harness/README.md), [experiment notebook](harness/analysis.ipynb) |
| **Agent evaluation** | A multimodal judge that uses trajectories, data inputs, and rendered charts to diagnose execution failures, data errors, chart errors, and readability problems. Analyzed the baseline against human labels and designed separate judge prompts for different failures. | [Judge](evaluation/agent_evaluation/validator/solution.py), [run guide](evaluation/README.md), [error analysis](evaluation/analysis.ipynb) |
| **Agent training** | Synchronous REINFORCE and Dr.GRPO, plus asynchronous Dr.GRPO with concurrent rollout workers, a bounded queue, policy versioning, and a staleness cutoff. Fine-tuned Qwen3.5-4B with LoRA on Number Search and TextCraft. | [Trainers](training/agent_training/train), [run guide](training/README.md), [experiment configs](training/configs) |

## Selected results

- **Context compaction.** On the recorded SWE-bench Django instance, peak prompt size dropped from **16,675 to 5,949 tokens**. The baseline made 14 model calls and the compaction run made 38. [Recorded trajectories](harness/artifacts)
- **Chart evaluation.** The included benchmark has **110 runs with human labels**. The evaluator reports per-category confusion counts and Matthews correlation, with execution failures scored separately from chart quality. [Labels](evaluation/seed_labels.json) · [Scoring code](evaluation/agent_evaluation/workflow/evaluate.py)
- **RL post-training.** In the reported experiments, Dr.GRPO fine-tuning improved single-attempt TextCraft success from **17% to 60%**. Asynchronous rollouts reduced time per training iteration by **23%** in a separate comparison. [TextCraft configuration](training/configs/textcraft_grpo.yaml) · [Asynchronous trainer](training/agent_training/train/async_grpo.py)

## Run locally

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Run each command from the repository root.

```bash
(cd harness && uv sync --locked && uv run --locked pytest)
(cd evaluation && uv sync --locked &&
  uv run --locked agent-eval validate-labels --labels seed_labels.json --artifacts artifacts)
(cd training && uv sync --locked && uv run --locked pytest)
```

These checks run locally. To import all three packages in a single activated environment, run `uv pip install ./harness ./evaluation ./training`.

## Set up Modal for live runs

Create a Modal account and choose a workspace. From `harness/` or `evaluation/`, install the locked dependencies and sign in.

```bash
uv sync --locked --python 3.12
uv run --locked modal setup
uv run --locked modal profile current
uv run --locked modal token info
```

Modal stores the CLI login locally so you can reuse it across all three components. If you start in `training/`, add `--extra modal` to both the `uv sync` and `uv run` commands because the Modal CLI is optional there. [Modal setup guide](https://modal.com/docs/guide/getting-started)

- **Agent harness.** From `harness/`, run `git submodule update --init -- chess_app` and copy `.env.example` to `.env`. Set `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` for your model provider. Run `uv run --locked agent-harness-doctor` before launching a sandbox. [Harness run guide](harness/README.md)
- **Agent evaluation.** From `evaluation/`, run `bash scripts/deploy_validator_model.sh` and copy `.env.example` to `.env`. Set `VALIDATOR_BASE_URL` to the deployed URL with `/v1` appended and leave `VALIDATOR_MODEL=validator`. Create a Modal proxy token in your workspace and set `VALIDATOR_API_KEY` to its combined `<token-id>.<token-secret>` value. For generating new trajectories, follow the guide for the model endpoints and agent runner. [Proxy tokens](https://modal.com/docs/guide/webhook-proxy-auth) · [Evaluation deployment guide](evaluation/infrastructure/README.md)
- **Agent training.** From `training/`, run `uv sync --locked --extra modal`. The SkyRL guide shows which launcher to deploy and how to set `TINKER_BASE_URL` to the exact URL printed by Modal. The Number Search profiles use two H100s, and the TextCraft profile uses four. [SkyRL on Modal](training/skyrl_tinker_modal/README.md#one-time-setup) · [Training commands](training/README.md)

## Attribution

Dongzhou Yu implemented the harness, evaluator, and training features described above while following Carnegie Mellon University's **11-768 AI Agents** course. The course team provided the starter infrastructure, environments, and benchmarks. The original histories and author attribution for all three repositories are preserved in this repository. Use `git log --graph --oneline HEAD` to view the implementation commits and the consolidation that follows them.
