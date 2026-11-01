"""Helpers for testing suites without a real agent."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from inspect_ai.model import ChatMessage

from runthrough.adapters.base import AgentTurn, TurnContext


class ScriptedTarget:
    """A fake agent that returns pre-written replies, in order.

    Useful for checking a suite's plumbing (scenarios, simulator, scorers) before
    pointing it at a real endpoint. Reply *n* answers turn *n* of every conversation;
    after the last reply, the last one repeats. Every call is recorded in ``calls``.
    """

    def __init__(
        self,
        replies: Sequence[AgentTurn | str],
        *,
        history: Literal["full", "session"] = "full",
    ) -> None:
        if not replies:
            raise ValueError("ScriptedTarget needs at least one reply")
        self.replies = [AgentTurn(text=r) if isinstance(r, str) else r for r in replies]
        self.history: Literal["full", "session"] = history
        self.calls: list[tuple[list[ChatMessage], TurnContext]] = []

    async def send(self, messages: list[ChatMessage], ctx: TurnContext) -> AgentTurn:
        index = min(ctx.turn - 1, len(self.replies) - 1)
        self.calls.append((list(messages), ctx))
        return self.replies[index]
