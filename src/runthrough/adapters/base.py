"""The contract every adapter implements.

An adapter sends the conversation so far to an agent endpoint and returns one
:class:`AgentTurn`: what the agent said, which tools it called, and what those
tools returned. The simulator solver turns that into Inspect chat messages.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from inspect_ai.model import ChatMessage
from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    """One tool call made by the agent, as observed in its response."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    error: str | None = None


class AgentTurn(BaseModel):
    """Everything the agent produced in reply to one user message."""

    text: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    latency_ms: int | None = None
    error: str | None = None
    raw_events: list[dict[str, Any]] = Field(default_factory=list)
    """The untouched response events, kept for debugging and custom scorers."""


class TurnContext(BaseModel):
    """Per-conversation values an adapter may need on every request."""

    conversation_id: str
    turn: int
    context: dict[str, Any] = Field(default_factory=dict)
    """The scenario's ``context`` values, passed through to the endpoint (e.g. profile data)."""


@runtime_checkable
class Target(Protocol):
    """An agent endpoint that runthrough can hold a conversation with."""

    @property
    def history(self) -> Literal["full", "session"]:
        """``"full"``: resend the whole conversation each turn (e.g. Vercel AI SDK).
        ``"session"``: the endpoint keeps history; send only the newest message."""
        ...

    async def send(self, messages: list[ChatMessage], ctx: TurnContext) -> AgentTurn: ...
