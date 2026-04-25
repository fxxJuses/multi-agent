"""
Minimal Plan-and-Execute agent — make a high-level plan first, then execute each
step with tool use, then synthesize the final answer. No agent framework.

Flow:
1. Planner emits a numbered step list for the task.
2. Executor works on one step at a time, optionally using tools in a ReAct-style loop.
3. Synthesizer combines all completed step results into the final answer.

Usage — OpenAI:
  export OPENAI_API_KEY=...
  export OPENAI_MODEL=gpt-4o-mini   # optional
  python plan_execute_agent.py

Usage — 火山引擎方舟（兼容 OpenAI SDK）:
  export ARK_API_KEY=<控制台 API Key>
  export ARK_MODEL=ep-xxxxxxxx
  export ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

Usage — 阿里云百炼 / DashScope（OpenAI 兼容模式）:
  export DASHSCOPE_API_KEY=sk-...
  export DASHSCOPE_MODEL=qwen-plus
  export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

  统一变量：LLM_MODEL 优先于各厂商的 *_MODEL。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from llm import LLM
from react_agent import TOOL_IMPL


PLANNER_DOCS = """
You are the Planner in a Plan-and-Execute agent.

Break the task into a short ordered plan.
Output only a numbered list, one step per line, like:
1. ...
2. ...
3. ...

Rules:
- Keep steps concrete and actionable.
- Use at most the requested number of steps.
- Do not solve the task yet.
- Do not include extra commentary or headings.
"""

EXECUTOR_DOCS = """
You are the Executor in a Plan-and-Execute agent.

You are working on exactly one current step. You may use tools if needed.

Output either:

Thought: ...
Action: <tool_name>
Action Input: <single-line input>

or:

Thought: ...
Step Answer: <result for this step>

Available tools:
- calculator: numeric expression with digits, + - * / ( )
- fake_search: short natural-language query (mock KB, not the real web)

Rules:
- Focus only on the current step.
- One Action per message.
- Wait for Observation before the next Action.
- If the step is complete, output Step Answer.
- Do not output Final Answer for the whole task.
"""

SYNTHESIZER_DOCS = """
You are the final synthesis module in a Plan-and-Execute agent.

You will receive:
- the original task
- the high-level plan
- the result of each completed step

Produce the final answer for the user directly and clearly.
If a step failed, use the successful steps and mention limits briefly.
"""

REPLAN_DOCS = """
You are the replanning module in a Plan-and-Execute agent.

Decide whether the remaining plan should continue as-is, should be replaced with
a new plan, or whether the task is already complete.

Output one of these formats only:

Decision: CONTINUE
Reason: <brief reason>

Decision: DONE
Reason: <brief reason>

Decision: REPLAN
Reason: <brief reason>
Updated Plan:
1. ...
2. ...

Rules:
- Prefer CONTINUE if the remaining plan still makes sense.
- Use DONE if the task is already sufficiently solved from completed steps.
- Use REPLAN only when the remaining steps are clearly no longer appropriate or
  important information suggests a better path.
- If you choose REPLAN, output only the remaining steps still needed from now on.
"""


@dataclass
class PlanStep:
    index: int
    text: str


@dataclass
class PlanExecuteAgent:
    """Hand-rolled Plan-and-Execute loop over the shared LLM wrapper."""

    model: str | None = None
    max_plan_steps: int = 8
    max_step_turns: int = 6
    max_replans: int = 2
    verbose: bool = False
    llm: LLM = field(init=False)

    def __post_init__(self) -> None:
        self.llm = LLM(model=self.model, system_prompt="")

    def _build_planner_prompt(self, user_task: str) -> str:
        return (
            PLANNER_DOCS.strip()
            + "\n\n"
            + f"Task:\n{user_task.strip()}\n\n"
            + f"Use at most {self.max_plan_steps} steps."
        )

    @staticmethod
    def _parse_plan(text: str) -> list[PlanStep]:
        steps: list[PlanStep] = []
        for raw_line in text.strip().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"(\d+)[\.\)]\s+(.+)", line)
            if not match:
                raise ValueError(f"invalid plan line: {line}")
            index = int(match.group(1))
            content = match.group(2).strip()
            steps.append(PlanStep(index=index, text=content))

        if not steps:
            raise ValueError("empty plan")
        return steps

    def _make_plan(self, user_task: str) -> tuple[list[PlanStep], str]:
        prompt = self._build_planner_prompt(user_task)
        repair_suffix = ""

        for _ in range(3):
            response = self.llm.chat(
                prompt + repair_suffix,
                system_prompt="You are a careful planning assistant.",
                temperature=0.1,
            ).content.strip()
            try:
                steps = self._parse_plan(response)
                if len(steps) > self.max_plan_steps:
                    raise ValueError(
                        f"plan has {len(steps)} steps, exceeds limit {self.max_plan_steps}"
                    )
                return steps, response
            except ValueError as exc:
                repair_suffix = (
                    "\n\nYour previous output was malformed.\n"
                    f"Error: {exc}\n"
                    "Rewrite the full numbered plan and follow the format exactly."
                )

        raise ValueError("planner failed to produce a valid plan after 3 attempts")

    def _build_executor_prompt(
        self,
        user_task: str,
        plan: list[PlanStep],
        finished_steps: list[tuple[PlanStep, str]],
        current_step: PlanStep,
        scratchpad: str,
    ) -> str:
        plan_text = "\n".join(f"{step.index}. {step.text}" for step in plan)
        completed_text = (
            "\n".join(f"{step.index}. {step.text}\nResult: {result}" for step, result in finished_steps)
            if finished_steps
            else "(none yet)"
        )
        return (
            EXECUTOR_DOCS.strip()
            + "\n\n"
            + f"Original task:\n{user_task.strip()}\n\n"
            + f"Full plan:\n{plan_text}\n\n"
            + f"Completed steps so far:\n{completed_text}\n\n"
            + f"Current step:\n{current_step.index}. {current_step.text}\n\n"
            + "Transcript for the current step (Thought / Action / Observation):\n"
            + scratchpad
        )

    @staticmethod
    def _parse_executor_output(text: str) -> dict[str, Any]:
        t = text.strip()

        step_answer = re.search(r"Step\s*Answer\s*:\s*(.+)", t, flags=re.IGNORECASE | re.DOTALL)
        if step_answer:
            return {"kind": "step_answer", "answer": step_answer.group(1).strip()}

        action = re.search(r"Action\s*:\s*(\S+)", t, flags=re.IGNORECASE)
        action_input = re.search(r"Action\s*Input\s*:\s*(.+)", t, flags=re.IGNORECASE | re.DOTALL)
        if action and action_input:
            name = action.group(1).strip().lower()
            inp = action_input.group(1).strip()
            inp = re.split(r"\n\s*(?:Thought|Action|Step Answer)\s*:", inp, flags=re.IGNORECASE)[0].strip()
            return {"kind": "action", "name": name, "input": inp}

        return {"kind": "parse_error", "raw": t}

    @staticmethod
    def _append_observation(scratchpad: str, observation: str) -> str:
        return scratchpad + f"\nObservation: {observation}\n"

    @staticmethod
    def _render_plan(plan: list[PlanStep]) -> str:
        return "\n".join(f"{step.index}. {step.text}" for step in plan)

    @staticmethod
    def _render_results(results: list[tuple[PlanStep, str]]) -> str:
        if not results:
            return "(none yet)"
        return "\n".join(f"{step.index}. {step.text}\nResult: {result}" for step, result in results)

    def _print_banner(self, title: str) -> None:
        if self.verbose:
            print(f"\n=== {title} ===")

    def _execute_step(
        self,
        user_task: str,
        plan: list[PlanStep],
        finished_steps: list[tuple[PlanStep, str]],
        current_step: PlanStep,
    ) -> str:
        scratchpad = ""

        for turn in range(1, self.max_step_turns + 1):
            prompt = self._build_executor_prompt(
                user_task=user_task,
                plan=plan,
                finished_steps=finished_steps,
                current_step=current_step,
                scratchpad=scratchpad,
            )
            assistant = self.llm.chat(
                prompt,
                system_prompt="You are a careful step executor.",
                temperature=0.2,
            ).content.strip()
            scratchpad += f"\n{assistant}\n"
            parsed = self._parse_executor_output(assistant)

            if parsed["kind"] == "step_answer":
                if self.verbose:
                    print(f"\n=== Execute {current_step.index}.{turn} ===")
                    print(assistant)
                return str(parsed["answer"])

            if parsed["kind"] == "parse_error":
                observation = (
                    "Parse error: expected 'Action:' and 'Action Input:' or 'Step Answer:'. "
                    "Please follow the format exactly."
                )
                if self.verbose:
                    print(f"\n=== Execute {current_step.index}.{turn} ===")
                    print(assistant)
                    print(f"Observation: {observation}")
                scratchpad = self._append_observation(scratchpad, observation)
                continue

            name = str(parsed["name"])
            inp = str(parsed["input"])
            tool_fn = TOOL_IMPL.get(name)
            if tool_fn is None:
                observation = f'Unknown tool "{name}". Use one of: {", ".join(TOOL_IMPL)}.'
            else:
                try:
                    observation = tool_fn(inp)
                except Exception as exc:  # noqa: BLE001 - tool boundary
                    observation = f"Error: {exc}"

            if self.verbose:
                print(f"\n=== Execute {current_step.index}.{turn} ===")
                print(assistant)
                print(f"Observation: {observation}")
            scratchpad = self._append_observation(scratchpad, observation)

        return f"Stopped: max_step_turns reached while executing step {current_step.index}."

    def _build_replan_prompt(
        self,
        user_task: str,
        current_plan: list[PlanStep],
        finished_steps: list[tuple[PlanStep, str]],
        remaining_steps: list[PlanStep],
    ) -> str:
        remaining_text = self._render_plan(remaining_steps) if remaining_steps else "(none left)"
        return (
            REPLAN_DOCS.strip()
            + "\n\n"
            + f"Task:\n{user_task.strip()}\n\n"
            + f"Current full plan:\n{self._render_plan(current_plan)}\n\n"
            + f"Completed step results:\n{self._render_results(finished_steps)}\n\n"
            + f"Remaining steps:\n{remaining_text}"
        )

    @staticmethod
    def _parse_replan_output(text: str) -> dict[str, Any]:
        t = text.strip()
        decision_match = re.search(r"Decision\s*:\s*(CONTINUE|REPLAN|DONE)", t, flags=re.IGNORECASE)
        reason_match = re.search(r"Reason\s*:\s*(.+)", t, flags=re.IGNORECASE)
        if not decision_match:
            return {"kind": "parse_error", "raw": t}

        decision = decision_match.group(1).upper()
        reason = reason_match.group(1).strip() if reason_match else ""
        if decision != "REPLAN":
            return {"kind": "decision", "decision": decision, "reason": reason}

        updated_match = re.search(r"Updated\s*Plan\s*:\s*(.+)", t, flags=re.IGNORECASE | re.DOTALL)
        if not updated_match:
            return {"kind": "parse_error", "raw": t}

        try:
            updated_steps = PlanExecuteAgent._parse_plan(updated_match.group(1).strip())
        except ValueError:
            return {"kind": "parse_error", "raw": t}
        return {
            "kind": "decision",
            "decision": decision,
            "reason": reason,
            "updated_steps": updated_steps,
        }

    def _should_replan(
        self,
        user_task: str,
        current_plan: list[PlanStep],
        finished_steps: list[tuple[PlanStep, str]],
        remaining_steps: list[PlanStep],
    ) -> dict[str, Any]:
        if not remaining_steps:
            return {"decision": "DONE", "reason": "No remaining steps."}

        prompt = self._build_replan_prompt(
            user_task=user_task,
            current_plan=current_plan,
            finished_steps=finished_steps,
            remaining_steps=remaining_steps,
        )
        for _ in range(3):
            response = self.llm.chat(
                prompt,
                system_prompt="You are a careful replanning assistant.",
                temperature=0.1,
            ).content.strip()
            parsed = self._parse_replan_output(response)
            if parsed["kind"] == "decision":
                parsed["raw"] = response
                return parsed
            prompt += (
                "\n\nYour previous output was malformed. Rewrite it using one of the allowed formats only."
            )

        return {
            "decision": "CONTINUE",
            "reason": "Fallback to continue because replanning output was malformed.",
            "raw": "",
        }

    def _synthesize(
        self,
        user_task: str,
        plan: list[PlanStep],
        results: list[tuple[PlanStep, str]],
    ) -> str:
        prompt = (
            SYNTHESIZER_DOCS.strip()
            + "\n\n"
            + f"Task:\n{user_task.strip()}\n\n"
            + f"Plan:\n{self._render_plan(plan)}\n\n"
            + f"Completed step results:\n{self._render_results(results)}"
        )
        return self.llm.chat(
            prompt,
            system_prompt="You are a careful answer synthesis assistant.",
            temperature=0.2,
        ).content.strip()

    def run(self, user_task: str) -> str:
        plan, raw_plan = self._make_plan(user_task)
        if self.verbose:
            self._print_banner("Planner Output")
            print(raw_plan)

        finished_steps: list[tuple[PlanStep, str]] = []
        active_plan = list(plan)
        replan_count = 0
        step_idx = 0

        while step_idx < len(active_plan):
            step = active_plan[step_idx]
            if self.verbose:
                self._print_banner(f"Current Step {step.index}")
                print(step.text)

            result = self._execute_step(
                user_task=user_task,
                plan=active_plan,
                finished_steps=finished_steps,
                current_step=step,
            )
            finished_steps.append((step, result))
            if self.verbose:
                self._print_banner(f"Step {step.index} Result")
                print(result)

            remaining_steps = active_plan[step_idx + 1 :]
            decision = self._should_replan(
                user_task=user_task,
                current_plan=active_plan,
                finished_steps=finished_steps,
                remaining_steps=remaining_steps,
            )
            if self.verbose:
                self._print_banner("Replan Decision")
                print(f"Decision: {decision['decision']}")
                if decision.get("reason"):
                    print(f"Reason: {decision['reason']}")
                if decision["decision"] == "REPLAN" and decision.get("updated_steps"):
                    print("Updated Plan:")
                    print(self._render_plan(decision["updated_steps"]))

            if decision["decision"] == "DONE":
                break
            if decision["decision"] == "REPLAN":
                if replan_count >= self.max_replans:
                    if self.verbose:
                        self._print_banner("Replan Skipped")
                        print("Reached max_replans; continuing current remaining plan.")
                else:
                    replan_count += 1
                    active_plan = finished_steps_to_plan(finished_steps) + decision["updated_steps"]
                    step_idx = len(finished_steps)
                    continue

            step_idx += 1

        answer = self._synthesize(user_task, active_plan, finished_steps)
        if self.verbose:
            self._print_banner("Final Synthesis")
            print(answer)
        return answer


def finished_steps_to_plan(results: list[tuple[PlanStep, str]]) -> list[PlanStep]:
    return [step for step, _ in results]


def main() -> None:
    verbose_value = (
        os.getenv("PLAN_EXECUTE_VERBOSE", "").strip()
        or os.getenv("REACT_VERBOSE", "").strip()
    )
    verbose = verbose_value.lower() in {"1", "true", "yes", "on"}
    agent = PlanExecuteAgent(verbose=verbose)
    demo = (
        "What is (128 * 7) / 2? Then say which is larger: that number or 500? "
        "Show the intermediate numeric result."
    )
    print("--- Task ---\n", demo, "\n")
    answer = agent.run(demo)
    print("--- Final Answer ---\n", answer, "\n")


if __name__ == "__main__":
    main()
