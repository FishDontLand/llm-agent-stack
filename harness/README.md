# Agent harness

`agent_harness` provides a shared ReAct loop for coding and chess agents. Each run records model prompts, tool observations, and compaction events. The coding agent emits a Git patch, and the chess agent uses HTTP tools for moves and game state. Optional skills expose tool instructions on demand.

## Configure a fresh checkout

Run these commands from `harness/`. Modal runs the sandboxes. Your model provider handles the chat completions separately.

```bash
uv sync --locked --python 3.12
uv run --locked pytest
git submodule update --init -- chess_app
uv run --locked modal setup
uv run --locked modal token info
cp .env.example .env
```

Edit `.env` using your inference provider's credentials.

| Variable | Value |
| --- | --- |
| `OPENAI_BASE_URL` | The HTTPS root of the provider's OpenAI-compatible API, such as `https://api.openai.com/v1`. |
| `OPENAI_API_KEY` | An API key from that provider. |
| `OPENAI_MODEL` | A provider model ID that supports Chat Completions with function tools and `reasoning_effort`. |
| `OPENAI_MAX_RETRIES` | Optional. The default is 5. |

Run `uv run --locked agent-harness-doctor` to check credentials, the model endpoint, and the pinned chess checkout before creating a sandbox. To test the Modal sandbox by itself, run `uv run --locked pytest -m modal tests/test_env.py`. This starts a billable sandbox and uses no model calls.

## Run the coding and chess agents

The commands below take the model from `.env`. The coding agent repairs the pinned chess app in a Modal sandbox, and the evaluator applies the resulting patch to a fresh testbed and runs the public tests.

```bash
output=artifacts/local-chess
uv run --locked agent-harness-code \
  --task tasks/chess-terminal-move --skills-path tasks/code-skills \
  --trajectory "$output/repair.json" --patch-output "$output/fix.patch"
uv run --locked python scripts/evaluate.py \
  --task tasks/chess-terminal-move --evaluation tasks/chess-terminal-move/public_tests \
  --patch "$output/fix.patch" -v
uv run --locked agent-harness-chess \
  --task tasks/chess-terminal-move --patch "$output/fix.patch" \
  --trajectory "$output/chess.json" --result "$output/game.json"
```

Check the evaluation exit status and test summary before playing. The chess trajectory records tool calls and observations. The game result records the final board state.

## Run the SWE-bench example

The bundled Django case runs in its published testbed image. The compaction threshold below is 6,000 estimated prompt tokens.

```bash
output=artifacts/local-swebench
uv run --locked agent-harness-swebench django__django-15368 \
  --skills-path tasks/code-skills --compact-threshold-tokens 6000 \
  --trajectory "$output/trajectory.json" --patch-output "$output/fix.patch"
uv run --locked python scripts/evaluate_swebench.py django__django-15368 \
  --patch "$output/fix.patch" -v
```

Omit `--compact-threshold-tokens` and choose another output directory for a full-context comparison. The [Makefile](Makefile) provides shortcuts with a default DeepSeek model ID. Pass `MODEL=your-provider-model-id` when using a different provider or model.

The live runs and patch checks use Modal compute. If you interrupt one, run `uv run --locked modal container list` to check for a leftover container and `uv run --locked modal container stop CONTAINER_ID` to stop it. [Modal container CLI](https://modal.com/docs/cli/latest/container)

The [analysis notebook](analysis.ipynb) compares prompt growth with and without compaction and reviews how legal-move observations affected chess tool calls.
