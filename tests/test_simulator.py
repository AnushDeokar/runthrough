from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    get_model,
)

from runthrough import AgentTurn, ToolCallRecord
from runthrough.scenario import Scenario, to_sample
from runthrough.simulator import (
    RESULT_KEY,
    STOP_TOKEN,
    simulated_conversation,
    simulator_view,
    turn_to_messages,
)
from runthrough.simulator.convert import clean_user_message
from runthrough.simulator.prompts import FIRST_TURN_MESSAGE, NO_REPLY_MESSAGE
from runthrough.testing import ScriptedTarget


def scenario(**overrides) -> Scenario:
    fields = {"id": "refund", "persona": "Priya, polite", "goal": "Refund a broken mug"}
    return Scenario.model_validate({**fields, **overrides})


def simulator_model(outputs: list[str] | Callable[[list[ChatMessage]], str]):
    """A mock simulator. Records every message list it was shown in ``seen``."""
    seen: list[list[ChatMessage]] = []
    queue = list(outputs) if isinstance(outputs, list) else None

    def respond(messages, *_):
        seen.append(list(messages))
        text = queue.pop(0) if queue is not None else outputs(messages)  # type: ignore[operator]
        return ModelOutput.from_content(model="mockllm/model", content=text)

    model = get_model("mockllm/model", custom_outputs=respond, memoize=False)
    return model, seen


def run(s: Scenario, target, simulator, tmp_path: Path, **kwargs) -> EvalSample:
    task = Task(
        dataset=MemoryDataset([to_sample(s)]),
        solver=simulated_conversation(target, simulator=simulator, **kwargs),
    )
    [log] = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    return _only_sample(log)


def _only_sample(log: EvalLog) -> EvalSample:
    assert log.samples is not None
    [sample] = log.samples
    return sample


def roles(sample: EvalSample) -> list[str]:
    return [m.role for m in sample.messages]


class TestConversation:
    def test_runs_until_simulator_stops(self, tmp_path: Path) -> None:
        model, seen = simulator_model(["My mug arrived broken", "Order 48213", STOP_TOKEN])
        target = ScriptedTarget(["Sorry! What's your order number?", "Refund issued."])

        sample = run(scenario(), target, model, tmp_path)

        assert [m.text for m in sample.messages] == [
            "My mug arrived broken",
            "Sorry! What's your order number?",
            "Order 48213",
            "Refund issued.",
        ]
        assert sample.metadata[RESULT_KEY]["stop_reason"] == "simulator_stopped"
        assert sample.metadata[RESULT_KEY]["turns"] == 2
        assert sample.output.completion == "Refund issued."
        assert len(seen) == 3

    def test_simulator_sees_persona_and_a_labelled_transcript(self, tmp_path: Path) -> None:
        model, seen = simulator_model(["Hi, my mug broke", STOP_TOKEN])
        run(scenario(), ScriptedTarget(["Sorry to hear that."]), model, tmp_path)

        first, second = seen
        assert "Priya, polite" in first[0].text and "Refund a broken mug" in first[0].text
        assert STOP_TOKEN in first[0].text
        assert [(m.role, m.text) for m in first[1:]] == [("user", FIRST_TURN_MESSAGE)]
        assert [m.role for m in second] == ["system", "user"]
        assert second[1].text == (
            "Conversation so far:\n\n"
            "You: Hi, my mug broke\n\n"
            "Assistant: Sorry to hear that.\n\n"
            "Write your next message."
        )

    def test_target_gets_full_history_context_and_one_conversation_id(self, tmp_path: Path) -> None:
        model, _ = simulator_model(["one", "two", STOP_TOKEN])
        target = ScriptedTarget(["a", "b"])
        sample = run(scenario(context={"profile": {"age": 4}}), target, model, tmp_path)

        (first_msgs, first_ctx), (second_msgs, second_ctx) = target.calls
        assert [m.text for m in first_msgs] == ["one"]
        assert [m.text for m in second_msgs] == ["one", "a", "two"]
        assert (first_ctx.turn, second_ctx.turn) == (1, 2)
        assert first_ctx.context == {"profile": {"age": 4}}
        assert first_ctx.conversation_id == second_ctx.conversation_id
        assert first_ctx.conversation_id == sample.metadata[RESULT_KEY]["conversation_id"]

    def test_session_target_gets_only_the_new_message(self, tmp_path: Path) -> None:
        model, _ = simulator_model(["one", "two", STOP_TOKEN])
        target = ScriptedTarget(["a", "b"], history="session")
        run(scenario(), target, model, tmp_path)
        assert [[m.text for m in msgs] for msgs, _ in target.calls] == [["one"], ["two"]]

    def test_opening_messages_are_sent_without_the_simulator(self, tmp_path: Path) -> None:
        model, seen = simulator_model(["Thanks", STOP_TOKEN])
        s = scenario(opening=["Hello", "My mug broke"])
        sample = run(s, ScriptedTarget(["Hi!", "Sorry!", "Bye"]), model, tmp_path)

        user_texts = [m.text for m in sample.messages if m.role == "user"]
        assert user_texts == ["Hello", "My mug broke", "Thanks"]
        assert len(seen) == 2

    def test_stops_at_max_turns(self, tmp_path: Path) -> None:
        model, seen = simulator_model(lambda _: "still here")
        sample = run(scenario(max_turns=3), ScriptedTarget(["ok"]), model, tmp_path)
        assert sample.metadata[RESULT_KEY] | {"conversation_id": None} == {
            "conversation_id": None,
            "turns": 3,
            "stop_reason": "max_turns",
        }
        assert len(seen) == 3

    def test_solver_max_turns_overrides_scenario(self, tmp_path: Path) -> None:
        model, _ = simulator_model(lambda _: "still here")
        sample = run(scenario(max_turns=8), ScriptedTarget(["ok"]), model, tmp_path, max_turns=2)
        assert sample.metadata[RESULT_KEY]["turns"] == 2

    def test_agent_error_ends_the_conversation(self, tmp_path: Path) -> None:
        model, _ = simulator_model(["hello", "anyone?"])
        target = ScriptedTarget([AgentTurn(text="", error="HTTP 500")])
        sample = run(scenario(), target, model, tmp_path)

        assert sample.metadata[RESULT_KEY]["stop_reason"] == "agent_error"
        assert len(target.calls) == 1
        assert sample.messages[-1].metadata == {"error": "HTTP 500"}

    def test_target_exception_becomes_sample_error(self, tmp_path: Path) -> None:
        class Broken:
            history: Literal["full", "session"] = "full"

            async def send(self, messages, ctx):
                raise ConnectionError("agent unreachable")

        model, _ = simulator_model(["hello"])
        task = Task(
            dataset=MemoryDataset([to_sample(scenario())]),
            solver=simulated_conversation(Broken(), simulator=model),
        )
        [log] = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
        assert log.status == "error"
        sample = _only_sample(log)
        assert sample.error is not None
        assert "agent unreachable" in sample.error.message

    def test_tool_calls_are_recorded_as_inspect_messages(self, tmp_path: Path) -> None:
        model, seen = simulator_model(["Can I talk to a person?", STOP_TOKEN])
        reply = AgentTurn(
            text="I've passed your request to our team.",
            tool_calls=[ToolCallRecord(id="c1", name="request_human_handoff", result={"ok": True})],
        )
        sample = run(scenario(), ScriptedTarget([reply]), model, tmp_path)

        assert roles(sample) == ["user", "assistant", "tool", "assistant"]
        call_message = sample.messages[1]
        assert isinstance(call_message, ChatMessageAssistant) and call_message.tool_calls
        assert call_message.tool_calls[0].function == "request_human_handoff"
        # The simulated user sees the reply text, never the tool call.
        assert seen[1][-1].text.endswith(
            "Assistant: I've passed your request to our team.\n\nWrite your next message."
        )
        assert "request_human_handoff" not in seen[1][-1].text

    def test_requires_a_runthrough_scenario(self, tmp_path: Path) -> None:
        from inspect_ai.dataset import Sample

        model, _ = simulator_model(["hi"])
        task = Task(
            dataset=MemoryDataset([Sample(input="plain")]),
            solver=simulated_conversation(ScriptedTarget(["a"]), simulator=model),
        )
        [log] = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
        sample = _only_sample(log)
        assert sample.error is not None and "no runthrough scenario" in sample.error.message


class TestSimulatorStopping:
    @pytest.mark.parametrize(
        "reply", [STOP_TOKEN, f"  {STOP_TOKEN}\n", f"Thanks! {STOP_TOKEN}", ""]
    )
    def test_stop_variants(self, reply: str, tmp_path: Path) -> None:
        model, _ = simulator_model([reply])
        target = ScriptedTarget(["unused"])
        sample = run(scenario(), target, model, tmp_path)
        assert target.calls == []
        assert sample.messages == []
        assert sample.metadata[RESULT_KEY]["stop_reason"] == "simulator_stopped"

    def test_prompt_must_have_placeholders(self) -> None:
        with pytest.raises(ValueError, match="stop_token"):
            simulated_conversation(ScriptedTarget(["a"]), prompt="{persona} {goal}")


class TestConversion:
    def test_text_only(self) -> None:
        [message] = turn_to_messages(AgentTurn(text="Hello", latency_ms=120))
        assert isinstance(message, ChatMessageAssistant)
        assert (message.text, message.metadata) == ("Hello", {"latency_ms": 120})

    def test_tool_call_with_error_and_no_text(self) -> None:
        turn = AgentTurn(
            text="",
            latency_ms=90,
            tool_calls=[
                ToolCallRecord(id="1", name="lookup", args={"q": "fees"}, result="AED 50,000"),
                ToolCallRecord(id="2", name="book_meeting", error="slot taken"),
            ],
        )
        call, ok, failed = turn_to_messages(turn)
        assert isinstance(call, ChatMessageAssistant) and call.tool_calls
        assert [c.arguments for c in call.tool_calls] == [{"q": "fees"}, {}]
        assert call.metadata == {"latency_ms": 90}
        assert isinstance(ok, ChatMessageTool) and ok.text == "AED 50,000" and ok.error is None
        assert isinstance(failed, ChatMessageTool) and failed.error is not None
        assert failed.error.message == "slot taken"

    def test_structured_tool_result_is_json(self) -> None:
        turn = AgentTurn(text="", tool_calls=[ToolCallRecord(id="1", name="t", result={"a": 1})])
        assert turn_to_messages(turn)[1].text == '{"a": 1}'

    def test_view_marks_missing_replies(self) -> None:
        conversation: list[ChatMessage] = [
            ChatMessageUser(content="hi"),
            ChatMessageAssistant(content=""),
            ChatMessageUser(content="hello?"),
        ]
        [_, prompt] = simulator_view("system", conversation)
        assert prompt.text == (
            "Conversation so far:\n\n"
            f"You: hi\n\nAssistant: {NO_REPLY_MESSAGE}\n\n"
            f"You: hello?\n\nAssistant: {NO_REPLY_MESSAGE}\n\n"
            "Write your next message."
        )

    @pytest.mark.parametrize(
        ("raw", "clean"),
        [
            ('"Hi there"', "Hi there"),
            ("User: hello", "hello"),
            ("You: hey", "hey"),
            ("  ok  ", "ok"),
            ("it's", "it's"),
        ],
    )
    def test_clean_user_message(self, raw: str, clean: str) -> None:
        assert clean_user_message(raw) == clean


def test_example_task_runs(tmp_path: Path) -> None:
    import importlib.util

    path = Path(__file__).parents[1] / "examples" / "suites" / "support" / "task.py"
    spec = importlib.util.spec_from_file_location("support_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model, _ = simulator_model(
        lambda view: STOP_TOKEN if "Assistant:" in view[-1].text else "Order 48213"
    )
    [log] = eval(
        module.support(),
        model=model,
        log_dir=str(tmp_path),
        display="none",
    )
    assert log.status == "success"
    assert log.samples is not None and len(log.samples) == 2
    for sample in log.samples:
        assert sample.metadata[RESULT_KEY]["stop_reason"] == "simulator_stopped"
        assert sample.messages[0].role == "user"


class TestFramingGuard:
    def test_echoed_transcript_is_retried(self, tmp_path: Path) -> None:
        echo = "Use the conversation history below.\n\nPrevious conversation:\nYou: hi"
        model, seen = simulator_model(["Hi there", echo, "Thanks, that helps", STOP_TOKEN])
        sample = run(scenario(), ScriptedTarget(["Hello!", "Sure."]), model, tmp_path)
        assert [m.text for m in sample.messages if m.role == "user"] == [
            "Hi there",
            "Thanks, that helps",
        ]
        assert "wasn't a chat message" in seen[2][-1].text

    def test_second_echo_is_cut_to_the_message(self, tmp_path: Path) -> None:
        model, _ = simulator_model(["Hi", "Assistant: Hello!", "Okay great\nYou: more", STOP_TOKEN])
        sample = run(scenario(), ScriptedTarget(["Hello!"]), model, tmp_path)
        assert [m.text for m in sample.messages if m.role == "user"] == ["Hi", "Okay great"]

    @pytest.mark.parametrize(
        ("text", "framing"),
        [
            ("Conversation so far:\n\nYou: hi", True),
            ("Sure. Assistant: hello", True),
            ("Can you help me with fees?", False),
            ("You: hi", False),  # a leading label is cleaned elsewhere
        ],
    )
    def test_looks_like_framing(self, text: str, framing: bool) -> None:
        from runthrough.simulator.convert import looks_like_framing

        assert looks_like_framing(text) is framing
