"""
Minimal ReAct (Reasoning + Acting) agent — loop built by hand, no agent framework.

Flow: LLM emits Thought / Action / Action Input → we run a local tool → append
Observation → repeat until the model emits Final Answer (or max steps).

Usage — OpenAI:
  export OPENAI_API_KEY=...
  export OPENAI_MODEL=gpt-4o-mini   # optional
  python react_agent.py

Usage — 火山引擎方舟（兼容 OpenAI SDK，见官方文档「兼容 OpenAI SDK」）:
  export ARK_API_KEY=<控制台 API Key>
  export ARK_MODEL=ep-xxxxxxxx        # 接入点 ID，不是「模型昵称」
  # 可选；若只设置了 ARK_API_KEY，则默认指向北京方舟 v3
  export ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

Usage — 阿里云百炼 / DashScope（OpenAI 兼容模式，见「如何通过 OpenAI 接口调用千问模型」）:
  export DASHSCOPE_API_KEY=sk-...
  export DASHSCOPE_MODEL=qwen-plus    # 或 LLM_MODEL；模型名见百炼文档
  # 可选；仅配置了 DASHSCOPE_API_KEY 时，默认北京 compatible-mode/v1
  export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
  # 其他地域示例：新加坡 dashscope-intl…、美东 dashscope-us…、香港 cn-hongkong.dashscope…（以官方文档为准）

  也可用 OPENAI_BASE_URL + OPENAI_API_KEY 指向上述 base_url（与 OpenAI 客户端一致）。

  统一变量：LLM_MODEL 优先于各厂商的 *_MODEL。
"""

from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from llm import LLM


# --- Tools (plain callables; names must match what the model outputs after "Action:") ---


def tool_calculator(expression: str) -> str:
    """Evaluate a numeric expression with + - * / and parentheses. No variables."""
    expr = expression.strip()

    allowed_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd)

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_ops:
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("division by zero")
                return left / right
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_ops:
            v = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +v
            if isinstance(node.op, ast.USub):
                return -v
        raise ValueError("unsupported expression")

    tree = ast.parse(expr, mode="eval")
    if not isinstance(tree, ast.Expression):
        raise ValueError("expected expression")
    result = _eval(tree.body)
    return str(int(result) if result == int(result) else result)


def tool_fake_search(query: str) -> str:
    """Tiny mock KB for demos (no real web)."""
    q = query.lower().strip()
    kb = {
        "capital of france": "Paris is the capital of France.",
        "capital of japan": "Tokyo is the capital of Japan.",
        "react paper": "ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022).",
    }
    for key, val in kb.items():
        if key in q:
            return val
    return f'No canned entry for "{query}". Try rephrasing or use calculator for math.'


TOOL_IMPL: dict[str, Callable[[str], str]] = {
    "calculator": tool_calculator,
    "fake_search": tool_fake_search,
}

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
        return (
            "You are a careful assistant using the ReAct pattern.\n"
            + TOOL_DOCS.strip()
        )

    def _build_prompt(self, user_task: str, scratchpad: str) -> str:
        return (
            f"Task:\n{user_task.strip()}\n\n"
            "Below is the transcript so far (Thought / Action / Observation). "
            "Continue with the next Thought (and Action if needed), or Final Answer.\n"
            f"{scratchpad}"
        )

    def _parse(self, text: str) -> dict[str, Any]:
        """Extract Final Answer or (Action, Action Input) from model output."""
        t = text.strip()

        # Final Answer: ... (take rest of string)
        m_final = re.search(r"Final\s*Answer\s*:\s*(.+)", t, flags=re.IGNORECASE | re.DOTALL)
        if m_final:
            return {"kind": "final", "answer": m_final.group(1).strip()}

        m_action = re.search(r"Action\s*:\s*(\S+)", t, flags=re.IGNORECASE)
        m_input = re.search(r"Action\s*Input\s*:\s*(.+)", t, flags=re.IGNORECASE | re.DOTALL)
        if m_action and m_input:
            name = m_action.group(1).strip().lower()
            inp = m_input.group(1).strip()
            # Stop at next structural line if model appended extra
            inp = re.split(r"\n\s*(?:Thought|Action|Final Answer)\s*:", inp, flags=re.IGNORECASE)[0].strip()
            return {"kind": "action", "name": name, "input": inp}

        return {"kind": "parse_error", "raw": t}

    def _append_observation(self, scratchpad: str, observation: str) -> str:
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
            inp = str(parsed["input"])
            fn = TOOL_IMPL.get(name)
            if fn is None:
                observation = f'Unknown tool "{name}". Use one of: {", ".join(TOOL_IMPL)}.'
                self._print_step(step, assistant, observation)
                scratchpad = self._append_observation(scratchpad, observation)
                continue
            try:
                obs = fn(inp)
            except Exception as e:  # noqa: BLE001 — demo tool boundary
                obs = f"Error: {e}"
            self._print_step(step, assistant, obs)
            scratchpad = self._append_observation(scratchpad, obs)

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
