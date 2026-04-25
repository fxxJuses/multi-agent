"""
Minimal ReAct (Reasoning + Acting) agent — loop built by hand, no agent framework.

Flow: LLM emits Thought / Action / Action Input → we run a local tool → append
Observation → repeat until the model emits Final Answer (or max steps).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from src.llm import LLM
from src.tools import TOOL_IMPL

TOOL_DOCS = """
You may call exactly one tool per turn by outputting:

Thought: ...your reasoning...
Action: <tool_name>
Action Input: <single-line input for the tool>

Available tools:
- calculator: Action Input is a math expression using digits, + - * / ( ).
- fake_search: Action Input is a short natural-language query (mock KB, not the real web).

When you have enough information to answer the user, output:

Thought: ...brief wrap-up...
Final Answer: ...answer in natural language for the user...

Rules:
- Do not invent Observation text; the user/system will append Observations after your Action.
- One Action per message; wait for Observation before the next Thought.
"""


@dataclass
class ReActAgent:
    """Hand-rolled ReAct loop over the shared LLM wrapper."""

    model: str | None = None
    max_steps: int = 12
    verbose: bool = False
    llm: LLM = field(init=False)

    def __post_init__(self) -> None:
        self.llm = LLM(model=self.model, system_prompt=self._system_prompt())

    @staticmethod
    def _system_prompt() -> str:
        return "You are a careful assistant using the ReAct pattern.\n" + TOOL_DOCS.strip()

    def _build_prompt(self, user_task: str, scratchpad: str) -> str:
        return (
            f"Task:\n{user_task.strip()}\n\n"
            "Below is the transcript so far (Thought / Action / Observation). "
            "Continue with the next Thought (and Action if needed), or Final Answer.\n"
            f"{scratchpad}"
        )

    def _parse(self, text: str) -> dict[str, Any]:
        t = text.strip()

        final_match = re.search(r"Final\s*Answer\s*:\s*(.+)", t, flags=re.IGNORECASE | re.DOTALL)
        if final_match:
            return {"kind": "final", "answer": final_match.group(1).strip()}

        action_match = re.search(r"Action\s*:\s*(\S+)", t, flags=re.IGNORECASE)
        input_match = re.search(r"Action\s*Input\s*:\s*(.+)", t, flags=re.IGNORECASE | re.DOTALL)
        if action_match and input_match:
            name = action_match.group(1).strip().lower()
            tool_input = input_match.group(1).strip()
            tool_input = re.split(
                r"\n\s*(?:Thought|Action|Final Answer)\s*:",
                tool_input,
                flags=re.IGNORECASE,
            )[0].strip()
            return {"kind": "action", "name": name, "input": tool_input}

        return {"kind": "parse_error", "raw": t}

    @staticmethod
    def _append_observation(scratchpad: str, observation: str) -> str:
        return scratchpad + f"\nObservation: {observation}\n"

    def _print_step(self, step: int, assistant: str, observation: str | None = None) -> None:
        if not self.verbose:
            return
        print(f"\n=== Step {step} ===")
        print(assistant)
        if observation is not None:
            print(f"Observation: {observation}")

    def run(self, user_task: str) -> str:
        scratchpad = ""
        for step in range(1, self.max_steps + 1):
            prompt = self._build_prompt(user_task, scratchpad)
            assistant = self.llm.chat(prompt, temperature=0.2).content.strip()
            scratchpad += f"\n{assistant}\n"

            parsed = self._parse(assistant)
            if parsed["kind"] == "final":
                self._print_step(step, assistant)
                return str(parsed["answer"])
            if parsed["kind"] == "parse_error":
                observation = (
                    "Parse error: expected 'Action:' and 'Action Input:' or 'Final Answer:'. "
                    "Please follow the format exactly."
                )
                self._print_step(step, assistant, observation)
                scratchpad = self._append_observation(scratchpad, observation)
                continue

            name = str(parsed["name"])
            tool_input = str(parsed["input"])
            tool_fn = TOOL_IMPL.get(name)
            if tool_fn is None:
                observation = f'Unknown tool "{name}". Use one of: {", ".join(TOOL_IMPL)}.'
                self._print_step(step, assistant, observation)
                scratchpad = self._append_observation(scratchpad, observation)
                continue

            try:
                observation = tool_fn(tool_input)
            except Exception as exc:  # noqa: BLE001 - demo tool boundary
                observation = f"Error: {exc}"
            self._print_step(step, assistant, observation)
            scratchpad = self._append_observation(scratchpad, observation)

        return "Stopped: max_steps reached without Final Answer."


def main() -> None:
    verbose = os.getenv("REACT_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
    agent = ReActAgent(verbose=verbose)
    demo = (
        "What is (128 * 7) / 2? Then say which is larger: that number or 500? "
        "Show the intermediate numeric result."
    )
    print("--- Task ---\n", demo, "\n")
    answer = agent.run(demo)
    print("--- Final Answer ---\n", answer, "\n")


if __name__ == "__main__":
    main()
