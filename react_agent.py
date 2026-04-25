"""Backward-compatible entrypoint for the refactored src layout."""

from src.agents.react_agent import ReActAgent, main
from src.tools.basic_tools import TOOL_IMPL, tool_calculator, tool_fake_search

__all__ = ["ReActAgent", "TOOL_IMPL", "tool_calculator", "tool_fake_search", "main"]


if __name__ == "__main__":
    main()
