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

## Quick start

Install Git and [uv](https://docs.astral.sh/uv/). These commands use Bash. On Windows, use WSL or Git Bash.

```bash
git clone https://github.com/FishDontLand/llm-agent-stack.git
cd llm-agent-stack
uv python install 3.12

(cd harness && uv sync --locked --python 3.12 && uv run --locked pytest)
(cd evaluation && uv sync --locked --python 3.12 &&
  uv run --locked agent-eval validate-labels --labels seed_labels.json --artifacts artifacts)
(cd training && uv sync --locked --python 3.12 && uv run --locked pytest)
```

The checks above use the bundled test data and need no cloud accounts or API keys. To import all three packages in a single activated environment, run `uv pip install ./harness ./evaluation ./training`.

## Environment setup for live runs

| Component | What to configure before running | From setup to first result |
| --- | --- | --- |
| **Agent harness** | Modal CLI login, an OpenAI-compatible endpoint that supports tool calling, and `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL` in `harness/.env`. Initialize the pinned `chess_app` submodule for chess tasks. | [Harness setup, patch verification, and chess run](harness/README.md) · [.env template](harness/.env.example) |
| **Agent evaluation** | Modal CLI login and a deployed multimodal vLLM validator, or an existing compatible endpoint. Set `VALIDATOR_BASE_URL`, `VALIDATOR_API_KEY`, and `VALIDATOR_MODEL` in `evaluation/.env`. Modal endpoints use a workspace proxy token. The generation models and agent runner are needed only for creating new trajectories. | [Judge a saved run and score the benchmark](evaluation/README.md) · [Deploy on Modal](evaluation/infrastructure/README.md) · [.env template](evaluation/.env.example) |
| **Agent training** | A Tinker-compatible backend that serves the model in the training YAML. For hosted Tinker, set `TINKER_API_KEY`. For SkyRL on Modal, install the `modal` extra, deploy the launcher that matches the YAML, and set `TINKER_BASE_URL` to the URL printed by Modal. The Number Search launchers request two H100s and the TextCraft launcher requests four. | [Training setup, short probe, and checkpoint evaluation](training/README.md) · [SkyRL deployment and cleanup](training/skyrl_tinker_modal/README.md) |

### Sign in to Modal

Create a Modal account and choose a workspace. From `harness/` or `evaluation/`, install the locked dependencies and sign in.

```bash
uv sync --locked --python 3.12
uv run --locked modal setup
uv run --locked modal profile current
uv run --locked modal token info
```

Modal stores the CLI login locally so you can reuse it across all three components. If you start in `training/`, add `--extra modal` to both the `uv sync` and `uv run` commands because the Modal CLI is optional there. [Modal setup guide](https://modal.com/docs/guide/getting-started)

## Attribution

Dongzhou Yu implemented the harness, evaluator, and training features described above while following Carnegie Mellon University's **11-768 AI Agents** course. The course team provided the starter infrastructure, environments, and benchmarks. The original histories and author attribution for all three repositories are preserved in this repository. Use `git log --graph --oneline HEAD` to view the implementation commits and the consolidation that follows them.
