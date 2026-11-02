import json
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import EvalLog
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Scorer
from inspect_ai.tool import ToolCall, ToolCallError

from runthrough import AgentTurn, ToolCallRecord, scenario_scorers, scenarios
from runthrough.scenario import Scenario, load_scenarios, to_sample
from runthrough.scorers import (
    expectation_scorers,
    no_agent_errors,
    parse_verdict,
    render_transcript,
    reply_language,
    rubric_judge,
    rubric_name,
    rubric_scorers,
    tool_order,
    tools_called,
    tools_not_called,
)
from runthrough.simulator import STOP_TOKEN, simulated_conversation
from runthrough.testing import ScriptedTarget

EXAMPLES = Path(__file__).parents[1] / "examples" / "suites" / "support" / "scenarios"

HANDOFF = AgentTurn(
    text="I've passed your request to our team.",
    tool_calls=[
        ToolCallRecord(id="1", name="lookup_order", args={"id": "48213"}, result={"ok": True}),
        ToolCallRecord(id="2", name="request_human_handoff", result={"success": False}),
    ],
)


def mock(outputs):
    seen = []
    queue = list(outputs) if isinstance(outputs, list) else None
    compute = None if isinstance(outputs, list) else outputs

    def respond(messages, *_):
        seen.append(list(messages))
        text = queue.pop(0) if queue is not None else compute(messages)  # type: ignore[misc]
        return ModelOutput.from_content(model="mockllm/model", content=text)

    return get_model("mockllm/model", custom_outputs=respond, memoize=False), seen


def run(
    scenario_list: list[Scenario],
    scorers: list[Scorer],
    tmp_path: Path,
    replies=("Hello there, how can I help you today?",),
) -> EvalLog:
    simulator, _ = mock(
        lambda view: STOP_TOKEN if "Assistant:" in view[-1].text else "Hi, I need help"
    )
    task = Task(
        dataset=MemoryDataset([to_sample(s) for s in scenario_list]),
        solver=simulated_conversation(ScriptedTarget(list(replies)), simulator=simulator),
        scorer=scorers,
    )
    [log] = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert log.status == "success", log.error
    return log


def scores(log: EvalLog, sample_id: str) -> dict:
    assert log.samples is not None
    sample = next(s for s in log.samples if s.id == sample_id)
    return {name: score for name, score in (sample.scores or {}).items()}


def metric(log: EvalLog, scorer_name: str) -> tuple[float, int]:
    assert log.results is not None
    result = next(s for s in log.results.scores if s.name == scorer_name)
    return result.metrics["accuracy"].value, result.scored_samples or 0


def scenario(id: str, **fields) -> Scenario:
    return Scenario.model_validate({"id": id, "persona": "p", "goal": "g", **fields})


class TestToolScorers:
    def test_called_not_called_and_order(self, tmp_path: Path) -> None:
        cases = [
            scenario(
                "pass",
                expect={
                    "tools_called": ["request_human_handoff"],
                    "tools_not_called": ["book"],
                    "tool_order": ["lookup_order", "request_human_handoff"],
                },
            ),
            scenario(
                "fail", expect={"tools_called": ["book"], "tools_not_called": ["lookup_order"]}
            ),
            scenario(
                "wrong-order", expect={"tool_order": ["request_human_handoff", "lookup_order"]}
            ),
            scenario("no-expectations"),
        ]
        scorers = [tools_called(), tools_not_called(), tool_order()]
        log = run(cases, scorers, tmp_path, replies=[HANDOFF])

        values = {i: {k: v.value for k, v in scores(log, i).items()} for i in [c.id for c in cases]}
        assert values == {
            "pass": {"tools_called": "C", "tools_not_called": "C", "tool_order": "C"},
            "fail": {"tools_called": "I", "tools_not_called": "I"},
            "wrong-order": {"tool_order": "I"},
            "no-expectations": {},
        }
        failed = scores(log, "fail")
        assert failed["tools_called"].explanation == (
            "Missing: book. Called: lookup_order, request_human_handoff."
        )
        assert (
            failed["tools_not_called"].explanation
            == "Called forbidden tools: lookup_order (turn 1)"
        )
        assert metric(log, "tools_called") == (0.5, 2)
        assert metric(log, "tool_order") == (0.5, 2)

    def test_no_agent_errors(self, tmp_path: Path) -> None:
        log = run([scenario("ok")], [no_agent_errors()], tmp_path)
        assert scores(log, "ok")["no_agent_errors"].value == "C"
        log = run(
            [scenario("broken")],
            [no_agent_errors()],
            tmp_path / "b",
            replies=[AgentTurn(text="", error="HTTP 500")],
        )
        score = scores(log, "broken")["no_agent_errors"]
        assert (score.value, score.explanation) == ("I", "HTTP 500")


class TestLanguage:
    def test_with_custom_detector(self, tmp_path: Path) -> None:
        detector = lambda text: "ar" if "مرحبا" in text else "en"  # noqa: E731
        cases = [scenario("ar", expect={"language": "ar"}), scenario("none")]
        log = run(
            cases,
            [reply_language(detector=detector)],
            tmp_path,
            replies=["Hello, how can I help you today?"],
        )
        score = scores(log, "ar")["reply_language"]
        assert score.value == "I" and score.explanation == "Expected ar; turn 1: en"
        assert scores(log, "none") == {}

    def test_langdetect_ignores_links_and_short_replies(self, tmp_path: Path) -> None:
        reply = "أهلاً! تخصص التصميم مدته ثلاث سنوات. [موقعنا](https://example.edu) +971 4 000 0000"
        log = run(
            [scenario("ar", expect={"language": "ar"})],
            [reply_language()],
            tmp_path,
            replies=[reply],
        )
        assert scores(log, "ar")["reply_language"].value == "C"

        log = run(
            [scenario("short", expect={"language": "ar"})],
            [reply_language()],
            tmp_path / "s",
            replies=["OK!"],
        )
        assert scores(log, "short") == {}


class TestJudge:
    @pytest.fixture
    def rubric(self, tmp_path: Path) -> Path:
        path = tmp_path / "Handoff Acknowledgement.md"
        path.write_text(
            "# Handoff\n\n- FAIL if the assistant promises a follow-up the tool did not confirm."
        )
        return path

    def test_pass_fail_and_applicability(self, rubric: Path, tmp_path: Path) -> None:
        verdict = {
            "evidence": "I've passed your request",
            "reasoning": "Tool said success false.",
            "verdict": "fail",
        }
        judge, seen = mock([json.dumps(verdict)])
        cases = [
            scenario("judged", rubrics=[str(rubric.resolve())], description="Handoff test"),
            scenario("not-judged"),
        ]
        log = run(cases, [rubric_judge(rubric, model=judge)], tmp_path / "log", replies=[HANDOFF])

        score = scores(log, "judged")["handoff_acknowledgement"]
        assert score.value == "I"
        assert (
            score.explanation
            == 'FAIL: Tool said success false.\nEvidence: "I\'ve passed your request"'
        )
        assert scores(log, "not-judged") == {}
        assert metric(log, "handoff_acknowledgement") == (0.0, 1)

        [[system, user]] = seen
        assert "Reply with a single JSON object" in system.text
        assert "promises a follow-up" in user.text and "Handoff test" in user.text
        assert '<- TOOL RESULT request_human_handoff: {"success": false}' in user.text

    def test_retry_then_noanswer(self, rubric: Path, tmp_path: Path) -> None:
        judge, seen = mock(["I think it passes.", '```json\n{"verdict": "PASS"}\n```'])
        s = scenario("x", rubrics=[str(rubric.resolve())])
        log = run([s], [rubric_judge(rubric, model=judge)], tmp_path / "a")
        assert scores(log, "x")["handoff_acknowledgement"].value == "C"
        assert len(seen) == 2

        judge, _ = mock(["no idea", "still no idea"])
        log = run([s], [rubric_judge(rubric, model=judge)], tmp_path / "b")
        score = scores(log, "x")["handoff_acknowledgement"]
        assert score.value == "N" and "could not be read" in (score.explanation or "")

    def test_rubric_scorers_from_dataset(self) -> None:
        names = {rubric_name(p) for s in load_scenarios(EXAMPLES) for p in s.rubrics}
        assert sorted(names) == ["claims_match_tools", "tone"]
        assert len(rubric_scorers(scenarios(EXAMPLES))) == 2
        assert len(scenario_scorers(scenarios(EXAMPLES))) == len(expectation_scorers()) + 2

    def test_duplicate_rubric_names(self, tmp_path: Path) -> None:
        for folder in ("a", "b"):
            (tmp_path / folder).mkdir()
            (tmp_path / folder / "tone.md").write_text("x")
        cases = [
            scenario("a", rubrics=[str(tmp_path / "a" / "tone.md")]),
            scenario("b", rubrics=[str(tmp_path / "b" / "tone.md")]),
        ]
        with pytest.raises(ValueError, match="share the name 'tone'"):
            rubric_scorers(cases)


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ('{"verdict": "pass"}', "pass"),
        ('Sure!\n```json\n{"reasoning": "ok", "verdict": "Fail"}\n```', "fail"),
        ('{"verdict": "maybe"}', None),
        ("pass", None),
    ],
)
def test_parse_verdict(text: str, verdict: str | None) -> None:
    result = parse_verdict(text)
    assert (result.verdict if result else None) == verdict


def test_render_transcript() -> None:
    messages = [
        ChatMessageUser(content="Book me in"),
        ChatMessageAssistant(
            content="Checking.",
            tool_calls=[ToolCall(id="1", function="book", arguments={"t": "10:00"})],
        ),
        ChatMessageTool(
            content="", tool_call_id="1", function="book", error=ToolCallError("unknown", "taken")
        ),
        ChatMessageTool(content="x" * 20, tool_call_id="2", function="lookup"),
        ChatMessageAssistant(content="", metadata={"error": "HTTP 500"}),
    ]
    assert render_transcript(messages, max_tool_chars=5) == "\n".join(
        [
            "[Turn 1] USER: Book me in",
            "ASSISTANT: Checking.",
            '  -> TOOL CALL book({"t": "10:00"})',
            "  <- TOOL ERROR book: taken",
            "  <- TOOL RESULT lookup: xxxxx ... [cut 15 characters]",
            "  !! AGENT ERROR: HTTP 500",
        ]
    )
