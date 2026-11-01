"""Render a conversation as plain text for an LLM judge."""

from __future__ import annotations

import json

from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageTool, ChatMessageUser


def render_transcript(messages: list[ChatMessage], *, max_tool_chars: int = 8000) -> str:
    """The conversation with tool calls and results, numbered by turn.

    Tool results longer than ``max_tool_chars`` are cut, keeping the start.
    """
    lines: list[str] = []
    turn = 0
    for message in messages:
        if isinstance(message, ChatMessageUser):
            turn += 1
            lines.append(f"\n[Turn {turn}] USER: {message.text.strip()}")
        elif isinstance(message, ChatMessageAssistant):
            if message.text.strip():
                lines.append(f"ASSISTANT: {message.text.strip()}")
            for call in message.tool_calls or []:
                args = json.dumps(call.arguments, ensure_ascii=False)
                lines.append(f"  -> TOOL CALL {call.function}({args})")
            error = (message.metadata or {}).get("error")
            if error:
                lines.append(f"  !! AGENT ERROR: {error}")
        elif isinstance(message, ChatMessageTool):
            if message.error is not None:
                lines.append(f"  <- TOOL ERROR {message.function}: {message.error.message}")
            else:
                lines.append(
                    f"  <- TOOL RESULT {message.function}: {_cut(message.text, max_tool_chars)}"
                )
    return "\n".join(lines).strip()


def _cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} ... [cut {len(text) - limit} characters]"
