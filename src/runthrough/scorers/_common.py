"""Helpers shared by runthrough's scorers."""

from __future__ import annotations

from dataclasses import dataclass

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser
from inspect_ai.solver import TaskState

from runthrough.scenario import Scenario, scenario_from_metadata


@dataclass(frozen=True)
class CalledTool:
    name: str
    turn: int
    """1-based index of the user message this call answered."""


def scenario_of(state: TaskState) -> Scenario:
    return scenario_from_metadata(state.metadata)


def called_tools(messages: list[ChatMessage]) -> list[CalledTool]:
    """Every tool call the agent made, in order, with the turn it happened in."""
    calls: list[CalledTool] = []
    turn = 0
    for message in messages:
        if isinstance(message, ChatMessageUser):
            turn += 1
        elif isinstance(message, ChatMessageAssistant):
            calls.extend(CalledTool(call.function, turn) for call in message.tool_calls or [])
    return calls


def agent_replies(messages: list[ChatMessage]) -> list[tuple[int, str]]:
    """The agent's non-empty text messages as ``(turn, text)``."""
    replies: list[tuple[int, str]] = []
    turn = 0
    for message in messages:
        if isinstance(message, ChatMessageUser):
            turn += 1
        elif isinstance(message, ChatMessageAssistant) and message.text.strip():
            replies.append((turn, message.text.strip()))
    return replies


def is_subsequence(expected: list[str], actual: list[str]) -> bool:
    """True if ``expected`` appears in ``actual`` in order, with anything in between."""
    remaining = iter(actual)
    return all(any(item == name for name in remaining) for item in expected)


def join(names: list[str]) -> str:
    return ", ".join(names) if names else "none"
