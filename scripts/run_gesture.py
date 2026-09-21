#!/usr/bin/env python3
"""Run the Gesture Agent with a simulated command or a synthetic frame."""

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.gesture.agent import GestureAgent
from src.agents.gesture.recognizer import MockGestureRecognizer
from src.agents.gesture.schemas import GestureAgentInput
from src.common.logger import setup_logger
from src.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run the HRI Gesture Agent.")
    parser.add_argument("--config", default=None, help="Path to gesture_config.yaml")
    parser.add_argument("--simulate", default=None, help="Gesture name such as STOP or POINT_LEFT")
    parser.add_argument("--frame-id", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("RunGesture")
    config = load_config(args.config) if args.config else {}

    recognizer = MockGestureRecognizer()
    agent = GestureAgent(config=config, recognizer=recognizer)
    agent.initialize()

    if args.simulate:
        agent_input = GestureAgentInput(
            gesture_override=args.simulate,
            frame_id=args.frame_id,
            source_name="simulation",
        )
    else:
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        agent_input = GestureAgentInput(frame=frame, frame_id=args.frame_id, source_name="synthetic")

    output = agent.process(agent_input)
    print(output.to_json(indent=2))
    logger.info(output.summary_text)
    agent.shutdown()


if __name__ == "__main__":
    main()