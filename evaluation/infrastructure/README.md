# Evaluation endpoints on Modal

These deployments serve the validator, generation models, and agent runner. Use the same Modal workspace and environment for all three deployments and the runner client.

## Deploy the validator

Run from `evaluation/`. To judge the bundled trajectories, deploy just the validator.

```bash
uv sync --locked --python 3.12
uv run --locked modal setup
uv run --locked modal token info
bash scripts/deploy_validator_model.sh
```

This app serves `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` on one L40S with the alias `validator`. Copy `.env.example` to `.env` and set `VALIDATOR_BASE_URL` to the URL printed for `ValidatorServer.serve` with `/v1` appended. Set `VALIDATOR_API_KEY` to a proxy token from that workspace in `<token-id>.<token-secret>` form. The CLI login authorizes deploying apps and the proxy token authenticates HTTP requests to them. [Create a proxy token](https://modal.com/docs/guide/webhook-proxy-auth) · [Run the judge](../README.md#configure-the-judge-endpoint)

## Generate new trajectories

Use these steps when you want the bundled agent runner to create new runs. The generation app deploys these independently scaling endpoints.

| Endpoint class | Model | Alias | GPU |
| --- | --- | --- | --- |
| `QwenServer` | `Qwen/Qwen2.5-Coder-3B-Instruct` | `qwen` | L4 |
| `MinistralServer` | `mistralai/Ministral-3-14B-Instruct-2512` | `ministral` | L40S |
| `GlmServer` | `zai-org/GLM-4.7-Flash` | `glm-4.7` | H100 |

```bash
bash scripts/deploy_generation_models.sh
```

Copy `.env.example` to `.env` if you have not done so already. Set `GENERATION_QWEN_BASE_URL`, `GENERATION_MINISTRAL_BASE_URL`, and `GENERATION_GLM_BASE_URL` to the URLs printed for `QwenServer.serve`, `MinistralServer.serve`, and `GlmServer.serve`, respectively, each with `/v1` appended. Set `GENERATION_API_KEY` to the proxy token in the same combined format.

```bash
bash scripts/deploy_agent_runner.sh
```

`deploy_agent_runner.sh` stores those settings in the private Modal secret
`agent-eval-generation-endpoints`. It deploys the gateway with the endpoint credentials and runs agent commands in separate containers.

From `evaluation/`, create a sample task and generate one trajectory using Qwen.

```bash
mkdir -p tasks/sample-chart
cat > tasks/sample-chart/task.json <<'JSON'
{
  "task_id": "sample-chart",
  "task_class": "bar-chart",
  "instructions": "Plot Apples: 3 and Oranges: 5 as a bar chart. Label the y-axis Count and save the figure as figure.png.",
  "inputs": []
}
JSON
uv run --locked agent-eval validate-tasks --tasks tasks
uv run --locked agent-eval generate --tasks tasks --models qwen --artifacts runs
```

The runner writes `runs/sample-chart__qwen/result.json` and saves `figure.png` there if the agent produces a figure. To judge generated runs, pass `--artifacts runs --artifact-root "$PWD"` to `agent-judge`. The first call to each generation endpoint starts its GPU container and can take several minutes.

## Stop the deployments

Use `uv run --locked modal app list` to see which apps you deployed. Stop the ones you used.

```bash
uv run --locked modal app stop agent-eval-validator-model
uv run --locked modal app stop agent-eval-agent-runner
uv run --locked modal app stop agent-eval-generation-gateway
uv run --locked modal app stop agent-eval-generation-models
```

The endpoints scale to zero after five idle minutes. Stopping their apps removes the endpoints until you redeploy. [Modal app CLI](https://modal.com/docs/cli/latest/app)
