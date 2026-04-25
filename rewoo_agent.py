"""Backward-compatible entrypoint for the refactored src layout."""

from src.agents.rewoo_agent import PlanStep, ReWOOAgent, main

__all__ = ["PlanStep", "ReWOOAgent", "main"]


if __name__ == "__main__":
    main()
