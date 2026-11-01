"""Simulated-user solver: drives a conversation between a simulator model and a target."""

from runthrough.simulator.convert import simulator_view, turn_to_messages
from runthrough.simulator.prompts import DEFAULT_SIMULATOR_PROMPT, STOP_TOKEN
from runthrough.simulator.solver import RESULT_KEY, simulated_conversation

__all__ = [
    "DEFAULT_SIMULATOR_PROMPT",
    "RESULT_KEY",
    "STOP_TOKEN",
    "simulated_conversation",
    "simulator_view",
    "turn_to_messages",
]
