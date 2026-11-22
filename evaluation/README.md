# Agent evaluation

`agent_evaluation` judges data-visualization runs using the task instructions, agent trajectory, data inputs, and rendered figure. It diagnoses execution failures first, then checks the chart's data, structure, and readability. Predictions carry an evidence string for every error.

## Check the bundled benchmark

Run commands from `evaluation/`. This check uses the recorded runs and human labels.

```bash
uv sync --locked --python 3.12
uv run --locked agent-eval validate-labels --labels seed_labels.json --artifacts artifacts
```

## Configure the judge endpoint

To deploy the included validator, sign in to a Modal workspace with access to an L40S GPU and run the deployment script.

```bash
uv run --locked modal setup
uv run --locked modal token info
bash scripts/deploy_validator_model.sh
cp .env.example .env
```

Create a proxy token in the same Modal workspace and set these entries in `.env`. If you already have a validator endpoint, use its URL, key, and model ID. The judge needs multimodal Chat Completions and vLLM-compatible `/tokenize` and `/detokenize` routes. [Modal proxy tokens](https://modal.com/docs/guide/webhook-proxy-auth)

| Variable | Value for the included deployment |
| --- | --- |
| `VALIDATOR_BASE_URL` | The exact URL printed for `ValidatorServer.serve`, with `/v1` appended. |
| `VALIDATOR_API_KEY` | Your Modal proxy token in `<token-id>.<token-secret>` form. |
| `VALIDATOR_MODEL` | `validator`, the alias set by the deployment. |

Check the endpoint on one saved run before judging the full benchmark. Keep the artifact root pointed at `evaluation/` so the recorded input paths still resolve. The first request can take several minutes while the model loads.

```bash
probe=$(mktemp -d)
cp -R artifacts/025 "$probe/"
uv run --locked agent-judge --artifacts "$probe" --artifact-root "$PWD" \
  --out "$probe/predictions.json"
cat "$probe/predictions.json"
```

## Judge and score the full benchmark

```bash
uv run --locked agent-judge --resume --out output/judge_predictions.json
uv run --locked agent-eval score --labels seed_labels.json --predictions output/judge_predictions.json
```

The scoring command compares predictions to human labels by run ID and reports per-family and macro Matthews correlation. To run the baseline for the analysis notebook, run `agent-judge` with `--solution agent_evaluation.validator.baseline --out output/baseline_predictions.json`. [Baseline error analysis](analysis.ipynb)

The generation models and runner are only needed when creating new trajectories. The deployment guide walks through configuring these endpoints, creating a sample task, and stopping the apps. [Evaluation deployment guide](infrastructure/README.md)
