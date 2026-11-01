"""Translate between runthrough's types and Inspect chat messages."""

from __future__ import annotations

import json

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall, ToolCallError

from runthrough.adapters.base import AgentTurn, ToolCallRecord
from runthrough.simulator.prompts import KICKOFF_MESSAGE, NO_REPLY_MESSAGE


def turn_to_messages(turn: AgentTurn) -> list[ChatMessage]:
    """Represent one agent turn as standard Inspect messages.

    Tool calls become an assistant message carrying ``tool_calls`` followed by one
    ``ChatMessageTool`` per result, so ``inspect view`` and scorers see them natively.
    The agent's text comes last.
    """
    metadata: dict[str, object] = {}
    if turn.latency_ms is not None:
        metadata["latency_ms"] = turn.latency_ms
    if turn.error is not None:
        metadata["error"] = turn.error
    needs_text_message = bool(turn.text) or not turn.tool_calls or turn.error is not None

    messages: list[ChatMessage] = []
    if turn.tool_calls:
        messages.append(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(id=call.id, function=call.name, arguments=call.args)
                    for call in turn.tool_calls
                ],
                metadata=None if needs_text_message else (metadata or None),
            )
        )
        messages.extend(_tool_message(call) for call in turn.tool_calls)
    if needs_text_message:
        messages.append(ChatMessageAssistant(content=turn.text, metadata=metadata or None))
    return messages


def simulator_view(system_prompt: str, conversation: list[ChatMessage]) -> list[ChatMessage]:
    """Build what the simulator model sees: the conversation with roles swapped.

    The simulated user's own messages become assistant messages and the agent's
    replies become user messages. Tool calls are hidden, because a real user never
    sees them.
    """
    view: list[ChatMessage] = [
        ChatMessageSystem(content=system_prompt),
        ChatMessageUser(content=KICKOFF_MESSAGE),
    ]
    agent_texts: list[str] | None = None
    for message in conversation:
        if isinstance(message, ChatMessageUser):
            if agent_texts is not None:
                view.append(_agent_reply(agent_texts))
            view.append(ChatMessageAssistant(content=message.text))
            agent_texts = []
        elif isinstance(message, ChatMessageAssistant) and agent_texts is not None:
            if message.text.strip():
                agent_texts.append(message.text.strip())
    if agent_texts is not None:
        view.append(_agent_reply(agent_texts))
    return view


def clean_user_message(text: str) -> str:
    """Remove wrapping that models sometimes add around a role-played message."""
    text = text.strip()
    for prefix in ("User:", "Me:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _agent_reply(texts: list[str]) -> ChatMessageUser:
    return ChatMessageUser(content="\n\n".join(texts) if texts else NO_REPLY_MESSAGE)


def _tool_message(call: ToolCallRecord) -> ChatMessageTool:
    if call.result is None:
        content = ""
    elif isinstance(call.result, str):
        content = call.result
    else:
        content = json.dumps(call.result, ensure_ascii=False, default=str)
    return ChatMessageTool(
        content=content,
        tool_call_id=call.id,
        function=call.name,
        error=ToolCallError("unknown", call.error) if call.error is not None else None,
    )
