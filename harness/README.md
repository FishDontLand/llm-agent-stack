# Agent harness

`agent_harness` provides a shared ReAct loop for coding and chess agents. Each run records model prompts, tool observations, and compaction events. The coding agent emits a Git patch, and the chess agent uses HTTP tools for moves and game state. Optional skills expose tool instructions on demand.

Run the examples from this directory.

```bash
uv sync --locked
uv run --locked pytest
uv run --locked agent-harness-code --help
uv run --locked agent-harness-swebench --help
uv run --locked agent-harness-chess --help
```

For live runs, copy `.env.example` to `.env`, configure an OpenAI-compatible model endpoint, and run `uv run modal setup`. Initialize the pinned chess testbed with `git submodule update --init -- chess_app` before trying `make run-code-agent` or `make run-chess-agent`. To run the SWE-bench example, use `make run-swebench-agent` with optional `MODEL`, `INSTANCE`, and `COMPACT_THRESHOLD` overrides.

The [analysis notebook](analysis.ipynb) compares prompt growth with and without compaction and reviews how legal-move observations affected chess tool calls.
