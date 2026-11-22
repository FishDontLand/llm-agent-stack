.PHONY: setup doctor verify-sources test test-modal test-chess-modal check-part1 check-swebench \
	run-code-agent run-swebench-agent run-chess-agent run-chess-agent-custom run-chess-agent-enhanced\
	run-obs-experiment-no-legal-moves run-obs-experiment-legal-moves \
	run-obs-deepseek-no-legal run-obs-deepseek-legal \
	run-obs-gpt-oss-no-legal run-obs-gpt-oss-legal \
	run-chess-modal

TASK ?= tasks/chess-terminal-move
CODE_SKILLS ?= tasks/code-skills
PATCH ?= artifacts/fix.patch
PART1_TRAJECTORY ?= artifacts/part1-trajectory.json
PUBLIC_EVAL ?= $(TASK)/public_tests
# A SWE-bench instance shipped with the harness.
INSTANCE ?= django__django-15368
SWEBENCH_PATCH ?= artifacts/$(INSTANCE).patch
SWEBENCH_TRAJECTORY ?= artifacts/$(INSTANCE)-trajectory.json
TRAJECTORY ?= artifacts/part3-trajectory.json
RESULT ?= artifacts/game-result.json
MODEL ?= deepseek/deepseek-v4-flash-0731
DEEPSEEK_MODEL ?= deepseek/deepseek-v4-flash-0731
GPT_OSS_MODEL ?= openai/gpt-oss-120b
MODEL_TAG ?= $(subst /,-,$(MODEL))
COMPACT_THRESHOLD ?= 6000
STEPS ?= 200
CHESS_TIMEOUT ?= 1800
OBS_NO_LEGAL_TRAJECTORY ?= artifacts/part3-no-legal-moves-$(MODEL_TAG).json
OBS_LEGAL_TRAJECTORY ?= artifacts/part3-legal-moves-$(MODEL_TAG).json
OBS_NO_LEGAL_RESULT ?= artifacts/part3-no-legal-moves-$(MODEL_TAG)-result.json
OBS_LEGAL_RESULT ?= artifacts/part3-legal-moves-$(MODEL_TAG)-result.json

setup:
	uv sync
	git submodule update --init -- chess_app
	$(MAKE) verify-sources

doctor:
	uv run agent-harness-doctor

verify-sources:
	uv run python -c "from agent_harness.task import Task; from agent_harness.utils.image import verify_source; verify_source(Task.load('tasks/chess-terminal-move'))"

# Offline harness tests.
test:
	uv run pytest

# Slow and billable: launches real Modal sandboxes.
test-modal:
	uv run pytest -m modal

test-chess-modal:
	uv run pytest -m modal tests/test_chess_sandbox.py

# Apply the generated patch and run the testbed regression suite.
check-part1:
	@test -f "$(PATCH)" || (echo "Patch not found: $(PATCH). Run make run-code-agent first."; exit 1)
	uv run python scripts/evaluate.py --task $(TASK) --evaluation $(PUBLIC_EVAL) \
		--patch $(PATCH) -v

# Apply a patch to the published SWE-bench image for INSTANCE and grade it
# against that instance's FAIL_TO_PASS and PASS_TO_PASS tests.
check-swebench:
	@test -f "$(SWEBENCH_PATCH)" || (echo "Patch not found: $(SWEBENCH_PATCH). Run make run-swebench-agent INSTANCE=$(INSTANCE) first."; exit 1)
	uv run python scripts/evaluate_swebench.py $(INSTANCE) --patch $(SWEBENCH_PATCH) -v

run-code-agent:
	uv run agent-harness-code --task $(TASK) --model $(MODEL) --step-limit $(STEPS) \
		--skills-path $(CODE_SKILLS) --trajectory $(PART1_TRAJECTORY) \
		--patch-output $(PATCH)

# Run the code agent on a vendored SWE-bench instance, in its published image.
run-swebench-agent:
	uv run agent-harness-swebench $(INSTANCE) --model $(MODEL) \
		--patch-output $(SWEBENCH_PATCH) \
		--trajectory $(SWEBENCH_TRAJECTORY) \
		--skills-path $(CODE_SKILLS) \
		--step-limit $(STEPS) $(if $(filter-out 0,$(COMPACT_THRESHOLD)),--compact-threshold-tokens $(COMPACT_THRESHOLD),)

run-chess-agent:
	uv run agent-harness-chess --model $(MODEL) --task $(TASK) --patch $(PATCH) \
		--step-limit $(STEPS) --sandbox-timeout $(CHESS_TIMEOUT) \
		--trajectory $(TRAJECTORY) --result $(RESULT)
	
run-chess-agent-custom:
	uv run agent-harness-chess --model $(MODEL) --task $(TASK) --patch $(PATCH) \
		--step-limit $(STEPS) --sandbox-timeout $(CHESS_TIMEOUT) \
		--trajectory $(TRAJECTORY) --result $(RESULT) --programmatic-tools

run-chess-agent-enhanced:
	uv run agent-harness-chess --model $(MODEL) --task $(TASK) --patch $(PATCH) \
		--step-limit $(STEPS) --sandbox-timeout $(CHESS_TIMEOUT) \
		--trajectory $(TRAJECTORY) --result $(RESULT) --programmatic-tools \
		--skills-path tasks/chess-skills --trajectory artifacts/part3-python-skill-trajectory.json

run-obs-experiment-no-legal-moves:
	uv run agent-harness-chess --model $(MODEL) --task $(TASK) --patch $(PATCH) \
		--omit-legal-moves --step-limit $(STEPS) --sandbox-timeout $(CHESS_TIMEOUT) \
		--trajectory $(OBS_NO_LEGAL_TRAJECTORY) --result $(OBS_NO_LEGAL_RESULT)

run-obs-experiment-legal-moves:
	uv run agent-harness-chess --model $(MODEL) --task $(TASK) --patch $(PATCH) \
		--step-limit $(STEPS) --sandbox-timeout $(CHESS_TIMEOUT) \
		--trajectory $(OBS_LEGAL_TRAJECTORY) --result $(OBS_LEGAL_RESULT)

run-obs-deepseek-no-legal:
	$(MAKE) run-obs-experiment-no-legal-moves MODEL="$(DEEPSEEK_MODEL)" MODEL_TAG=deepseek

run-obs-deepseek-legal:
	$(MAKE) run-obs-experiment-legal-moves MODEL="$(DEEPSEEK_MODEL)" MODEL_TAG=deepseek

run-obs-gpt-oss-no-legal:
	$(MAKE) run-obs-experiment-no-legal-moves MODEL="$(GPT_OSS_MODEL)" MODEL_TAG=gpt-oss

run-obs-gpt-oss-legal:
	$(MAKE) run-obs-experiment-legal-moves MODEL="$(GPT_OSS_MODEL)" MODEL_TAG=gpt-oss

run-chess-modal:
	uv run agent-harness-chess-modal --task $(TASK) --sandbox-timeout $(CHESS_TIMEOUT) $(if $(wildcard $(PATCH)),--patch $(PATCH),)