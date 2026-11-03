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


class AgentStep(BaseModel):
    """One model step inside a turn: optional text, then the tools it called."""

    text: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class AgentTurn(BaseModel):
    """Everything the agent produced in reply to one user message.

    Simple adapters fill ``text`` and ``tool_calls``. Adapters that can see the
    agent's individual steps also fill ``steps`` (use :meth:`from_steps`), which
    keeps text and tool calls in the order they happened.
    """

    text: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)
    message_id: str | None = None
    """The agent's own id for its reply, if it reports one."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    """Metadata the agent reported about its reply, e.g. model and token usage."""
    latency_ms: int | None = None
    error: str | None = None
    raw_events: list[dict[str, Any]] = Field(default_factory=list)
    """The untouched response events, kept for debugging and custom scorers."""

    @classmethod
    def from_steps(cls, steps: list[AgentStep], **fields: Any) -> AgentTurn:
        """Build a turn from steps, deriving ``text`` and ``tool_calls`` from them."""
        return cls(
            text="\n\n".join(step.text for step in steps if step.text),
            tool_calls=[call for step in steps for call in step.tool_calls],
            steps=steps,
            **fields,
        )


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
