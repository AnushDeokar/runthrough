"""Adapter for agents served with the Vercel AI SDK (v5) UI-message stream.

This is the protocol a ``useChat`` frontend speaks: the client POSTs
``{"id": <chat id>, "messages": [<UIMessage>...], ...}`` and the server streams
server-sent events such as ``text-delta``, ``tool-input-available`` and
``tool-output-available``. Protocol reference: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from typing import Any, Literal

import httpx
from httpx_sse import EventSource
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
)

from runthrough.adapters.base import AgentStep, AgentTurn, ToolCallRecord, TurnContext

RESERVED_BODY_KEYS = frozenset({"id", "messages", "trigger"})

Fields = Mapping[str, Any] | Callable[[TurnContext], Mapping[str, Any]]

# Deltas are already folded into text and tool inputs, so they are not kept in raw_events.
_DELTA_EVENTS = frozenset({"text-delta", "reasoning-delta", "tool-input-delta"})


class AdapterError(RuntimeError):
    """The endpoint could not be used: wrong URL, rejected credentials, or an unexpected
    response format. These are configuration problems, so the sample fails instead of
    being scored."""


def ai_sdk_endpoint(
    url: str,
    *,
    headers: Fields | None = None,
    body: Fields | None = None,
    timeout: float = 120.0,
) -> AISDKTarget:
    """Talk to an agent that streams the Vercel AI SDK UI-message protocol.

    Args:
        url: The chat route, e.g. ``http://localhost:3000/api/chat``.
        headers: Extra request headers, or a function of the turn context that
            returns them (useful for per-run ids).
        body: Extra top-level request fields. Defaults to the scenario's ``context``,
            the same way ``useChat`` sends its ``body`` option. ``id``, ``messages``
            and ``trigger`` are set by runthrough.
        timeout: Seconds to wait for the whole reply.
    """
    return AISDKTarget(url, headers=headers, body=body, timeout=timeout)


class AISDKTarget:
    """A :class:`~runthrough.adapters.base.Target` for AI SDK UI-message stream endpoints."""

    history: Literal["full", "session"] = "full"

    def __init__(
        self,
        url: str,
        *,
        headers: Fields | None = None,
        body: Fields | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.url = url
        self.headers = headers
        self.body = body
        self.timeout = timeout

    def build_request(self, messages: list[ChatMessage], ctx: TurnContext) -> dict[str, Any]:
        """The JSON body sent for one turn."""
        extra = dict(_resolve(self.body, ctx) if self.body is not None else ctx.context)
        clash = sorted(RESERVED_BODY_KEYS & extra.keys())
        if clash:
            raise AdapterError(f"Request body fields {clash} are set by runthrough; rename them.")
        return {
            **extra,
            "id": ctx.conversation_id,
            "messages": to_ui_messages(messages),
            "trigger": "submit-message",
        }

    async def send(self, messages: list[ChatMessage], ctx: TurnContext) -> AgentTurn:
        request_body = self.build_request(messages, ctx)
        headers = {"accept": "text/event-stream", **_resolve(self.headers, ctx)}
        started = time.perf_counter()

        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client,
            client.stream("POST", self.url, json=request_body, headers=headers) as response,
        ):
            if response.status_code >= 400:
                detail = (await response.aread()).decode(errors="replace")[:500]
                message = f"HTTP {response.status_code} from {self.url}: {detail}"
                if response.status_code >= 500:
                    return AgentTurn(text="", error=message, latency_ms=_ms_since(started))
                raise AdapterError(message)

            content_type = response.headers.get("content-type", "")
            if "text/event-stream" not in content_type:
                detail = (await response.aread()).decode(errors="replace")[:300]
                raise AdapterError(
                    f"{self.url} answered with content-type '{content_type}', not an AI SDK "
                    f"event stream. Check the URL points at the chat route. Body: {detail}"
                )

            events: list[dict[str, Any]] = []
            async for sse in EventSource(response).aiter_sse():
                if sse.data == "[DONE]":
                    break
                if not sse.data:
                    continue
                try:
                    events.append(json.loads(sse.data))
                except json.JSONDecodeError as exc:
                    raise AdapterError(
                        f"{self.url} sent an event that is not JSON: {sse.data[:200]}"
                    ) from exc

        return parse_ui_message_stream(events, latency_ms=_ms_since(started))


def parse_ui_message_stream(
    events: list[dict[str, Any]], *, latency_ms: int | None = None
) -> AgentTurn:
    """Fold UI-message stream events into one :class:`AgentTurn`."""
    steps: list[AgentStep] = []
    current: AgentStep | None = None
    open_text: dict[str, list[str]] = {}
    calls: dict[str, ToolCallRecord] = {}
    message_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = {}

    def step() -> AgentStep:
        nonlocal current
        if current is None:
            current = AgentStep()
            steps.append(current)
        return current

    def close_text(block_id: str) -> None:
        text = "".join(open_text.pop(block_id, []))
        if text:
            target = step()
            target.text = f"{target.text}\n\n{text}" if target.text else text

    for event in events:
        kind = event.get("type")
        if isinstance(event.get("messageMetadata"), dict):
            metadata.update(event["messageMetadata"])
        if kind == "start":
            message_id = event.get("messageId") or message_id
        elif kind == "start-step":
            current = None
            step()
        elif kind == "finish-step":
            for block_id in list(open_text):
                close_text(block_id)
            current = None
        elif kind == "text-start":
            open_text.setdefault(event["id"], [])
        elif kind == "text-delta":
            open_text.setdefault(event["id"], []).append(event.get("delta", ""))
        elif kind == "text-end":
            close_text(event["id"])
        elif kind in ("tool-input-available", "tool-input-error"):
            call = ToolCallRecord(
                id=event["toolCallId"],
                name=event["toolName"],
                args=_as_args(event.get("input")),
                error=event.get("errorText") if kind == "tool-input-error" else None,
            )
            calls[call.id] = call
            step().tool_calls.append(call)
        elif kind == "tool-output-available":
            if event["toolCallId"] in calls:
                calls[event["toolCallId"]].result = event.get("output")
        elif kind == "tool-output-error":
            if event["toolCallId"] in calls:
                calls[event["toolCallId"]].error = event.get("errorText") or "Tool failed"
        elif kind == "error":
            error = event.get("errorText") or "The agent reported an error"
        elif kind == "abort":
            error = error or "The agent aborted the response"

    for block_id in list(open_text):
        close_text(block_id)

    return AgentTurn.from_steps(
        steps,
        message_id=message_id,
        error=error,
        metadata=metadata,
        latency_ms=latency_ms,
        raw_events=[e for e in events if e.get("type") not in _DELTA_EVENTS],
    )


def to_ui_messages(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Rebuild the conversation as AI SDK v5 UIMessages, as ``useChat`` would resend it.

    Each agent turn (assistant and tool messages between two user messages) becomes
    one assistant UIMessage whose parts hold its steps, text and tool results.
    """
    results = {m.tool_call_id: m for m in messages if isinstance(m, ChatMessageTool)}
    ui: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] | None = None
    assistant_id: str | None = None

    def flush() -> None:
        nonlocal parts, assistant_id
        if parts:
            ui.append(
                {"id": assistant_id or f"assistant-{len(ui)}", "role": "assistant", "parts": parts}
            )
        parts, assistant_id = None, None

    for message in messages:
        if isinstance(message, ChatMessageUser):
            flush()
            ui.append(
                {
                    "id": message.id,
                    "role": "user",
                    "parts": [{"type": "text", "text": message.text}],
                }
            )
        elif isinstance(message, ChatMessageAssistant):
            if parts is None:
                parts = []
            assistant_id = (message.metadata or {}).get("message_id") or assistant_id
            step_parts: list[dict[str, Any]] = []
            if message.text:
                step_parts.append({"type": "text", "text": message.text, "state": "done"})
            for call in message.tool_calls or []:
                step_parts.append(
                    _tool_part(call.id, call.function, call.arguments, results.get(call.id))
                )
            if step_parts:
                parts.append({"type": "step-start"})
                parts.extend(step_parts)
    flush()
    return ui


def _tool_part(
    call_id: str, name: str, args: dict[str, Any], result: ChatMessageTool | None
) -> dict[str, Any]:
    part: dict[str, Any] = {"type": f"tool-{name}", "toolCallId": call_id, "input": args}
    if result is None:
        part["state"] = "input-available"
    elif result.error is not None:
        part.update(state="output-error", errorText=result.error.message)
    else:
        part.update(state="output-available", output=_decode(result.text))
    return part


def _decode(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _as_args(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {} if value is None else {"input": value}


def _resolve(fields: Fields | None, ctx: TurnContext) -> Mapping[str, Any]:
    if fields is None:
        return {}
    return fields(ctx) if callable(fields) else fields


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
