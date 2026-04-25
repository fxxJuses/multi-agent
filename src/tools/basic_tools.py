from __future__ import annotations

import ast
from typing import Callable


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
            value = _eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, ast.USub):
                return -value
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
    for key, value in kb.items():
        if key in q:
            return value
    return f'No canned entry for "{query}". Try rephrasing or use calculator for math.'


TOOL_IMPL: dict[str, Callable[[str], str]] = {
    "calculator": tool_calculator,
    "fake_search": tool_fake_search,
}
