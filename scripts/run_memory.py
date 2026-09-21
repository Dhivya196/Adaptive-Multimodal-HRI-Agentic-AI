#!/usr/bin/env python3
"""Run the in-memory HRI context agent from the command line."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.memory.agent import MemoryAgent
from src.agents.memory.schemas import MemoryAgentInput, MemoryOperation


def parse_args():
    parser = argparse.ArgumentParser(description="Run the HRI Memory Agent.")
    parser.add_argument("--config", default=None, help="Path to memory_config.yaml")
    parser.add_argument("--action", default="pick_and_place")
    parser.add_argument("--target", default="bottle")
    parser.add_argument("--location", default="LEFT")
    parser.add_argument("--query", default=None)
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    agent = MemoryAgent(config={})
    agent.initialize()

    if args.clear:
        output = agent.process(MemoryAgentInput(operation=MemoryOperation.CLEAR))
    elif args.query:
        output = agent.process(
            MemoryAgentInput(operation=MemoryOperation.RETRIEVE, query=args.query)
        )
    else:
        output = agent.process({
            "action": args.action,
            "target": args.target,
            "location": args.location,
            "task_status": "VALID",
        })

    print(output.to_json(indent=2))
    agent.shutdown()


if __name__ == "__main__":
    main()