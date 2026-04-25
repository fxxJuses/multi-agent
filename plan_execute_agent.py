"""Backward-compatible entrypoint for the refactored src layout."""

from src.agents.plan_execute_agent import PlanExecuteAgent, PlanStep, main

__all__ = ["PlanExecuteAgent", "PlanStep", "main"]


if __name__ == "__main__":
    main()
