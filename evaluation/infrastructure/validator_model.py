"""Deploy the multimodal validator endpoint on Modal.

Run `uv run modal deploy infrastructure/validator_model.py` from the evaluation directory.
"""

from __future__ import annotations

import modal
import serving

MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

app = modal.App("agent-eval-validator-model")


@app.cls(
    image=serving.base_image(),
    gpu="L40S:1",
    max_containers=1,
    scaledown_window=5 * serving.MINUTES,
    timeout=10 * serving.MINUTES,
    volumes=serving.volumes(),
)
@modal.concurrent(max_inputs=16)
class ValidatorServer:
    @modal.enter()
    def start(self) -> None:
        self.process = serving.start(MODEL, "validator", [
            "--tool-call-parser", "hermes",
            "--enable-auto-tool-choice",
            "--max-num-seqs", "2",
            "--max-model-len", "32768",
            "--max-num-batched-tokens", "32768",
        ])

    @modal.web_server(port=serving.PORT, startup_timeout=10 * serving.MINUTES,
                      requires_proxy_auth=True)
    def serve(self) -> None:
        pass

    @modal.exit()
    def stop(self) -> None:
        serving.stop(self.process)
