"""
Minimal ReWOO (Reasoning WithOut Observation) agent — plan first, then execute
tools, then synthesize the final answer. No agent framework.

Flow:
1. Planner emits a full plan with evidence variables (#E1, #E2, ...).
2. Executor runs the referenced tools, substituting prior evidence variables.
3. Solver answers the user using the original task, plan, and gathered evidence.

Usage — OpenAI:
  export OPENAI_API_KEY=...
  export OPENAI_MODEL=gpt-4o-mini   # optional
  python rewoo_agent.py

Usage — 火山引擎方舟（兼容 OpenAI SDK，见官方文档「兼容 OpenAI SDK」）:
  export ARK_API_KEY=<控制台 API Key>
  export ARK_MODEL=ep-xxxxxxxx        # 接入点 ID，不是「模型昵称」
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

from llm import LLM
from react_agent import TOOL_IMPL


PLANNER_TOOL_DOCS = """
You are the Planner in a ReWOO agent.

Create a complete tool-use plan before seeing any tool outputs.
Use this exact format:

Plan: <brief reason for this step>
#E1 = <tool_name>[<tool input>]
Plan: <brief reason for next step>
#E2 = <tool_name>[<tool input, may reference earlier evidence like #E1>]

Available tools:
- calculator: numeric expression with digits, + - * / ( )
- fake_search: short natural-language query (mock KB, not the real web)

Rules:
- Output only Plan / #E lines, no extra commentary.
- Every evidence line must use a unique variable like #E1, #E2, #E3.
- Tool names must be one of: calculator, fake_search.
- Tool inputs may reference earlier evidence variables.
- Do not produce Final Answer in the planning phase.
"""

SOLVER_DOCS = """
You are the Solver in a ReWOO agent.

You will receive:
- the original task
- the executed plan
- the evidence collected for each variable

Write the final answer for the user directly and clearly.
If some evidence contains errors, use the successful evidence and mention limits briefly.
"""


@dataclass
class PlanStep:
    plan: str
    evidence: str
    tool: str
    tool_input: str


@dataclass
class ReWOOAgent:
    """Hand-rolled ReWOO loop over the shared LLM wrapper."""

    model: str | None = None
    max_plan_steps: int = 8
    verbose: bool = False
    llm: LLM = field(init=False)

    def __post_init__(self) -> None:
        self.llm = LLM(model=self.model, system_prompt="")

    def _build_planner_prompt(self, user_task: str) -> str:
        return (
            PLANNER_TOOL_DOCS.strip()
            + "\n\n"
            + f"Task:\n{user_task.strip()}\n\n"
            + f"Limit the plan to at most {self.max_plan_steps} evidence steps."
        )

    @staticmethod
    def _parse_plan(text: str) -> list[PlanStep]:
        lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
        steps: list[PlanStep] = []
        pending_plan: str | None = None
        seen_evidence: set[str] = set()

        for line in lines:
            if line.lower().startswith("plan:"):
                pending_plan = line.split(":", 1)[1].strip()
                continue

            match = re.match(r"(#E\d+)\s*=\s*([a-zA-Z_][\w-]*)\[(.*)\]\s*$", line)
            if not match:
                raise ValueError(f"invalid plan line: {line}")
            if not pending_plan:
                raise ValueError(f"missing Plan before evidence line: {line}")

            evidence, tool, tool_input = match.groups()
            tool = tool.strip().lower()
            if evidence in seen_evidence:
                raise ValueError(f"duplicate evidence variable: {evidence}")

            steps.append(
                PlanStep(
                    plan=pending_plan,
                    evidence=evidence,
                    tool=tool,
                    tool_input=tool_input.strip(),
                )
            )
            seen_evidence.add(evidence)
            pending_plan = None

        if pending_plan:
            raise ValueError("dangling Plan without matching evidence line")
        if not steps:
            raise ValueError("empty plan")
        return steps

    def _make_plan(self, user_task: str) -> tuple[list[PlanStep], str]:
        prompt = self._build_planner_prompt(user_task)
        repair_suffix = ""

        for attempt in range(3):
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
                    "Rewrite the full plan and follow the format exactly."
                )

        raise ValueError("planner failed to produce a valid ReWOO plan after 3 attempts")

    @staticmethod
    def _substitute_vars(text: str, evidence_map: dict[str, str]) -> str:
        resolved = text
        for name in sorted(evidence_map, key=len, reverse=True):
            resolved = resolved.replace(name, evidence_map[name])
        return resolved

    def _execute(self, steps: list[PlanStep]) -> dict[str, str]:
        evidence_map: dict[str, str] = {}

        for idx, step in enumerate(steps, start=1):
            tool_input = self._substitute_vars(step.tool_input, evidence_map)
            tool_fn = TOOL_IMPL.get(step.tool)
            if tool_fn is None:
                result = f'Error: unknown tool "{step.tool}"'
            else:
                try:
                    result = tool_fn(tool_input)
                except Exception as exc:  # noqa: BLE001 - tool boundary
                    result = f"Error: {exc}"

            evidence_map[step.evidence] = result

            if self.verbose:
                print(f"\n=== Execute {idx} ===")
                print(f"Plan: {step.plan}")
                print(f"{step.evidence} = {step.tool}[{tool_input}]")
                print(f"Observation: {result}")

        return evidence_map

    @staticmethod
    def _render_plan(steps: list[PlanStep], evidence_map: dict[str, str] | None = None) -> str:
        rendered: list[str] = []
        for step in steps:
            line = f"Plan: {step.plan}\n{step.evidence} = {step.tool}[{step.tool_input}]"
            if evidence_map is not None:
                line += f"\n{step.evidence} -> {evidence_map.get(step.evidence, '')}"
            rendered.append(line)
        return "\n".join(rendered)

    def _solve(self, user_task: str, steps: list[PlanStep], evidence_map: dict[str, str]) -> str:
        prompt = (
            SOLVER_DOCS.strip()
            + "\n\n"
            + f"Task:\n{user_task.strip()}\n\n"
            + "Executed plan and evidence:\n"
            + self._render_plan(steps, evidence_map)
        )
        return self.llm.chat(
            prompt,
            system_prompt="You are a careful answer synthesis assistant.",
            temperature=0.2,
        ).content.strip()

    def run(self, user_task: str) -> str:
        steps, raw_plan = self._make_plan(user_task)
        if self.verbose:
            print("\n=== Planner Output ===")
            print(raw_plan)

        evidence_map = self._execute(steps)
        answer = self._solve(user_task, steps, evidence_map)

        if self.verbose:
            print("\n=== Solver Output ===")
            print(answer)
        return answer


def main() -> None:
    verbose = os.getenv("REWOO_VERBOSE", "").strip().lower() in {"1", "true", "yes", "on"}
    agent = ReWOOAgent(verbose=verbose)
    demo = (
        "What is (128 * 7) / 2? Then say which is larger: that number or 500? "
        "Show the intermediate numeric result."
    )
    print("--- Task ---\n", demo, "\n")
    answer = agent.run(demo)
    print("--- Final Answer ---\n", answer, "\n")


if __name__ == "__main__":
    main()
