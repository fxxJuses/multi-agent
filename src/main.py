from __future__ import annotations

import argparse

from src.agents.plan_execute_agent import PlanExecuteAgent
from src.agents.react_agent import ReActAgent
from src.agents.rewoo_agent import ReWOOAgent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one of the agent demos.")
    parser.add_argument(
        "--agent",
        choices=["react", "rewoo", "plan_execute"],
        default="react",
        help="Agent pattern to run.",
    )
    parser.add_argument(
        "--task",
        default=(
            "What is (128 * 7) / 2? Then say which is larger: that number or 500? "
            "Show the intermediate numeric result."
        ),
        help="Task to solve.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print intermediate planning and execution details.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    agent_map = {
        "react": ReActAgent,
        "rewoo": ReWOOAgent,
        "plan_execute": PlanExecuteAgent,
    }
    agent = agent_map[args.agent](verbose=args.verbose)
    print("--- Task ---\n", args.task, "\n")
    answer = agent.run(args.task)
    print("--- Final Answer ---\n", answer, "\n")


if __name__ == "__main__":
    main()
