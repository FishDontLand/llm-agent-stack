"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Any

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment

class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools += [EXECUTE_TOOL, SEND_MESSAGE_TOOL]

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        
        self.system_prompt = f"""You are an experienced software developer and your are responsible for coding tasks on the following system.
        <system_information>
        {{
            "machine": {self.env.machine},
            "release": {self.env.release},
            "system": {self.env.system},
            "version": {self.env.version}
        }}
        </system_information>"""

        self.task_prompt = "Below is a detailed description of the task. \n"
        self.task_prompt += task

        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        if self.skills:
            skill_prompt = [v['metadata'] for v in self.skills.values()]
            skill_prompt = "\n".join(skill_prompt)
            self.system_prompt += '\n Available Skills: \n' + skill_prompt

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        result = []
        for call in tool_calls:
            call_func = call.get("function", {})
            func_name = call_func.get("name", "unknown")
            content = None
            try:
                arguments = json.loads(call_func.get("arguments"))
            except Exception as e:
                content = f"Invalid arguments: {e}"
            
            if content is None:
                if func_name == "execute":
                    content = self.env.execute(
                        command=arguments.get("command", ""),
                        timeout=arguments.get("timeout"),
                        cwd=arguments.get("cwd"),
                        env=arguments.get("env"),
                        shell=arguments.get("shell", True)
                    )
                    content = format_tool_output(content)
                elif func_name == "send_message":
                    content = arguments.get("summary", "")
                    self.finished = True

                elif func_name == "invoke_skill":
                    skill_name = arguments.get("name")
                    if skill_name is None:
                        content = "skill name is missing"
                    else:
                        content = self.skills.get(skill_name, {}).get('content', '')
                else:
                    content = f"Unknown function: {func_name}"

            result.append({
                'role': 'tool',
                'content': content,
                'tool_call_id': call.get('id', ""),
            })

        return result
