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

from runthrough.adapters.base import AgentStep, AgentTurn, ToolCallRecord
from runthrough.simulator.prompts import (
    AGENT_LABEL,
    FIRST_TURN_MESSAGE,
    NEXT_TURN_INSTRUCTION,
    NO_REPLY_MESSAGE,
    USER_LABEL,
)


def turn_to_messages(turn: AgentTurn) -> list[ChatMessage]:
    """Represent one agent turn as standard Inspect messages.

    Each step becomes an assistant message (its text plus any ``tool_calls``),
    followed by one ``ChatMessageTool`` per tool result, so ``inspect view`` and
    scorers see the agent's work natively. Latency, errors and the agent's message
    id go in the metadata of the turn's last assistant message.
    """
    metadata: dict[str, object] = {}
    if turn.latency_ms is not None:
        metadata["latency_ms"] = turn.latency_ms
    if turn.message_id is not None:
        metadata["message_id"] = turn.message_id
    if turn.error is not None:
        metadata["error"] = turn.error

    messages: list[ChatMessage] = []
    last_assistant: ChatMessageAssistant | None = None
    for step in turn.steps or _steps_of(turn):
        if not step.text and not step.tool_calls:
            continue
        last_assistant = ChatMessageAssistant(
            content=step.text,
            tool_calls=[
                ToolCall(id=call.id, function=call.name, arguments=call.args)
                for call in step.tool_calls
            ]
            or None,
        )
        messages.append(last_assistant)
        messages.extend(_tool_message(call) for call in step.tool_calls)

    if last_assistant is None:
        last_assistant = ChatMessageAssistant(content="")
        messages.append(last_assistant)
    if metadata:
        last_assistant.metadata = metadata
    return messages


def _steps_of(turn: AgentTurn) -> list[AgentStep]:
    """Steps for a turn that only has ``text`` and ``tool_calls``: tools first, then text."""
    return [AgentStep(tool_calls=turn.tool_calls), AgentStep(text=turn.text)]


def simulator_view(system_prompt: str, conversation: list[ChatMessage]) -> list[ChatMessage]:
    """Build what the simulator model sees.

    A system prompt, then one user message holding the conversation so far as a
    labelled transcript ("You:" for the simulated user, "Assistant:" for the agent).
    Tool calls are left out, because a real user never sees them. Keeping the
    transcript in one message means the simulator's model calls read naturally in
    ``inspect view``, with no swapped roles.
    """
    lines: list[str] = []
    agent_texts: list[str] | None = None
    for message in conversation:
        if isinstance(message, ChatMessageUser):
            if agent_texts is not None:
                lines.append(_agent_line(agent_texts))
            lines.append(f"{USER_LABEL}: {message.text.strip()}")
            agent_texts = []
        elif isinstance(message, ChatMessageAssistant) and agent_texts is not None:
            if message.text.strip():
                agent_texts.append(message.text.strip())
    if agent_texts is not None:
        lines.append(_agent_line(agent_texts))

    if lines:
        prompt = "Conversation so far:\n\n" + "\n\n".join(lines) + f"\n\n{NEXT_TURN_INSTRUCTION}"
    else:
        prompt = FIRST_TURN_MESSAGE
    return [ChatMessageSystem(content=system_prompt), ChatMessageUser(content=prompt)]


def clean_user_message(text: str) -> str:
    """Remove wrapping that models sometimes add around a role-played message."""
    text = text.strip()
    for prefix in (f"{USER_LABEL}:", "User:", "Me:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _agent_line(texts: list[str]) -> str:
    return f"{AGENT_LABEL}: " + ("\n\n".join(texts) if texts else NO_REPLY_MESSAGE)


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
