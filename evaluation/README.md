# Agent evaluation

`agent_evaluation` judges data-visualization runs using the task instructions, agent trajectory, data inputs, and rendered figure. It diagnoses execution failures first, then checks the chart's data, structure, and readability. Predictions carry an evidence string for every error.

The recorded benchmark and labels can be validated offline. Run from this directory.

```bash
uv sync --locked
uv run --locked agent-eval validate-labels --labels seed_labels.json --artifacts artifacts
```

To produce predictions, configure `VALIDATOR_BASE_URL`, `VALIDATOR_API_KEY`, and `VALIDATOR_MODEL` using `.env.example`. The judge uses an OpenAI-compatible multimodal endpoint with vLLM-compatible tokenizer routes. The [deployment guide](infrastructure/README.md) covers Modal.

```bash
uv run --locked agent-judge --resume --out output/judge_predictions.json
uv run --locked agent-eval score --labels seed_labels.json --predictions output/judge_predictions.json
```

To reproduce the [baseline error analysis](analysis.ipynb), run `agent-judge` with `--solution agent_evaluation.validator.baseline --out output/baseline_predictions.json`. The scoring command compares predictions to human labels by run ID and reports per-family and macro Matthews correlation.
