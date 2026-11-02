import json
from pathlib import Path

import httpx
import pytest
import respx
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    get_model,
)

from runthrough import AgentStep, ToolCallRecord, TurnContext, ai_sdk_endpoint
from runthrough.adapters import AdapterError
from runthrough.adapters.ai_sdk import parse_ui_message_stream, to_ui_messages
from runthrough.scenario import Scenario, to_sample
from runthrough.simulator import RESULT_KEY, STOP_TOKEN, simulated_conversation

FIXTURES = Path(__file__).parent / "fixtures" / "ai_sdk"
URL = "http://agent.test/api/chat"
SSE = {"content-type": "text/event-stream", "x-vercel-ai-ui-message-stream": "v1"}


def fixture(name: str) -> str:
    return (FIXTURES / f"{name}.sse").read_text()


def events(name: str) -> list[dict]:
    lines = [line.removeprefix("data: ") for line in fixture(name).splitlines()]
    return [json.loads(line) for line in lines if line and line != "[DONE]"]


def ctx(**context) -> TurnContext:
    return TurnContext(
        conversation_id="5f0c8a4e-0000-4000-8000-000000000001", turn=1, context=context
    )


class TestParsing:
    def test_text_only(self) -> None:
        turn = parse_ui_message_stream(events("text_only"))
        assert turn.text == "Hello! How can I help you with admissions today?"
        assert (turn.tool_calls, turn.error, turn.message_id) == ([], None, "msg-assistant-1")
        assert [e["type"] for e in turn.raw_events].count("text-delta") == 0

    def test_tool_then_text(self) -> None:
        turn = parse_ui_message_stream(events("tool_then_text"))
        assert turn.steps == [
            AgentStep(
                tool_calls=[
                    ToolCallRecord(
                        id="call_r1",
                        name="retrieval",
                        args={"query": "FS1 fees"},
                        result={
                            "query": "FS1 fees",
                            "chunks": [
                                {
                                    "text": "FS1 annual fees are AED 52,000.",
                                    "source": "fees-2026.pdf",
                                }
                            ],
                        },
                    )
                ]
            ),
            AgentStep(text="Annual fees for FS1 are AED 52,000."),
        ]
        assert turn.text == "Annual fees for FS1 are AED 52,000."

    def test_text_and_failed_tool_in_one_step(self) -> None:
        turn = parse_ui_message_stream(events("two_steps_tool_error"))
        assert [s.text for s in turn.steps] == [
            "",
            "Let me book 10:00 for you.",
            "Sorry, that slot was just taken. Would 14:30 work?",
        ]
        assert [c.name for c in turn.tool_calls] == ["check_meeting_availability", "book_meeting"]
        booking = turn.steps[1].tool_calls[0]
        assert booking.args == {"date": "2026-11-03", "time": "10:00"}
        assert (booking.result, booking.error) == (None, "An error occurred.")
        assert turn.error is None

    def test_streamed_tool_input_and_reasoning(self) -> None:
        turn = parse_ui_message_stream(events("streamed_tool_input_with_reasoning"))
        [call] = turn.tool_calls
        assert (call.name, call.args, call.result) == (
            "request_human_handoff",
            {"reason": "asked for admissions"},
            {"notified": True},
        )
        assert turn.text == "I've passed your request to our admissions team."

    def test_stream_error(self) -> None:
        turn = parse_ui_message_stream(events("model_error"))
        assert (turn.text, turn.error) == ("", "An error occurred.")

    def test_stream_without_steps_or_text_end(self) -> None:
        turn = parse_ui_message_stream(
            [
                {"type": "text-delta", "id": "a", "delta": "Hi"},
                {"type": "text-delta", "id": "a", "delta": "!"},
            ]
        )
        assert turn.text == "Hi!"

    def test_abort(self) -> None:
        assert (
            parse_ui_message_stream([{"type": "abort"}]).error == "The agent aborted the response"
        )


class TestRequest:
    def test_first_turn(self) -> None:
        body = ai_sdk_endpoint(URL).build_request(
            [ChatMessageUser(content="Hi", id="u1")],
            ctx(profileData={"age": 4}, promptKey="system_prompt_ar"),
        )
        assert body == {
            "id": "5f0c8a4e-0000-4000-8000-000000000001",
            "messages": [{"id": "u1", "role": "user", "parts": [{"type": "text", "text": "Hi"}]}],
            "trigger": "submit-message",
            "profileData": {"age": 4},
            "promptKey": "system_prompt_ar",
        }

    def test_history_rebuilds_assistant_parts(self) -> None:
        turn = parse_ui_message_stream(events("two_steps_tool_error"))
        from runthrough.simulator import turn_to_messages

        history = [ChatMessageUser(content="Book me in", id="u1"), *turn_to_messages(turn)]
        history.append(ChatMessageUser(content="14:30 is fine", id="u2"))
        messages = ai_sdk_endpoint(URL).build_request(history, ctx())["messages"]

        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assistant = messages[1]
        assert assistant["id"] == "msg-assistant-1"
        assert [p["type"] for p in assistant["parts"]] == [
            "step-start",
            "tool-check_meeting_availability",
            "step-start",
            "text",
            "tool-book_meeting",
            "step-start",
            "text",
        ]
        availability, booking = assistant["parts"][1], assistant["parts"][4]
        assert availability["state"] == "output-available"
        assert availability["output"] == {"date": "2026-11-03", "slots": ["10:00", "14:30"]}
        assert booking == {
            "type": "tool-book_meeting",
            "toolCallId": "call_b1",
            "input": {"date": "2026-11-03", "time": "10:00"},
            "state": "output-error",
            "errorText": "An error occurred.",
        }

    def test_orphan_tool_message_and_empty_assistant(self) -> None:
        history = [
            ChatMessageUser(content="fees?", id="u1"),
            ChatMessageAssistant(content="", tool_calls=[]),
            ChatMessageTool(content="AED 52,000", tool_call_id="x", function="lookup"),
        ]
        # A tool message without a matching call is ignored; an empty assistant adds no message.
        assert [m["role"] for m in to_ui_messages(history)] == ["user"]

    def test_body_override_and_reserved_keys(self) -> None:
        target = ai_sdk_endpoint(URL, body=lambda c: {"chatTurn": c.turn})
        assert target.build_request([], ctx(ignored=True))["chatTurn"] == 1
        with pytest.raises(AdapterError, match="'messages'"):
            ai_sdk_endpoint(URL, body={"messages": []}).build_request([], ctx())


class TestHttp:
    @respx.mock
    async def test_send_parses_stream_and_sends_headers(self) -> None:
        route = respx.post(URL).mock(
            return_value=httpx.Response(200, headers=SSE, text=fixture("tool_then_text"))
        )
        target = ai_sdk_endpoint(URL, headers=lambda c: {"x-api-key": "k", "x-turn": str(c.turn)})

        turn = await target.send([ChatMessageUser(content="fees?")], ctx())

        assert turn.text == "Annual fees for FS1 are AED 52,000."
        assert turn.latency_ms is not None
        request = route.calls.last.request
        assert (request.headers["x-api-key"], request.headers["x-turn"]) == ("k", "1")
        assert request.headers["accept"] == "text/event-stream"
        assert json.loads(request.content)["id"] == "5f0c8a4e-0000-4000-8000-000000000001"

    @respx.mock
    async def test_server_error_becomes_agent_error(self) -> None:
        respx.post(URL).mock(return_value=httpx.Response(500, text="boom"))
        turn = await ai_sdk_endpoint(URL).send([ChatMessageUser(content="hi")], ctx())
        assert turn.error == f"HTTP 500 from {URL}: boom"

    @respx.mock
    async def test_client_error_raises(self) -> None:
        respx.post(URL).mock(return_value=httpx.Response(401, json={"error": "bad key"}))
        with pytest.raises(AdapterError, match="HTTP 401"):
            await ai_sdk_endpoint(URL).send([ChatMessageUser(content="hi")], ctx())

    @respx.mock
    async def test_non_stream_response_raises(self) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
        with pytest.raises(AdapterError, match="not an AI SDK event stream"):
            await ai_sdk_endpoint(URL).send([ChatMessageUser(content="hi")], ctx())

    @respx.mock
    async def test_non_json_event_raises(self) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, headers=SSE, text="data: hello\n\n"))
        with pytest.raises(AdapterError, match="not JSON"):
            await ai_sdk_endpoint(URL).send([ChatMessageUser(content="hi")], ctx())


@respx.mock
def test_end_to_end_with_simulator(tmp_path: Path) -> None:
    replies = iter([fixture("two_steps_tool_error"), fixture("text_only")])
    route = respx.post(URL).mock(
        side_effect=lambda request: httpx.Response(200, headers=SSE, text=next(replies))
    )
    script = iter(["Can I book a visit on the 3rd?", "14:30 works", STOP_TOKEN])
    simulator = get_model(
        "mockllm/model",
        custom_outputs=lambda *_: ModelOutput.from_content(
            model="mockllm/model", content=next(script)
        ),
        memoize=False,
    )
    scenario = Scenario(
        id="visit", persona="Parent", goal="Book a visit", context={"promptKey": "p1"}
    )
    task = Task(
        dataset=MemoryDataset([to_sample(scenario)]),
        solver=simulated_conversation(ai_sdk_endpoint(URL), simulator=simulator),
    )

    [log] = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")

    assert log.samples is not None
    [sample] = log.samples
    assert sample.metadata[RESULT_KEY]["stop_reason"] == "simulator_stopped"
    assert [m.role for m in sample.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
    first, second = (json.loads(call.request.content) for call in route.calls)
    assert first["id"] == second["id"] == sample.metadata[RESULT_KEY]["conversation_id"]
    assert first["promptKey"] == "p1"
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
