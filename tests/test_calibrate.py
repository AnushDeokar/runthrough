import json
import threading
from pathlib import Path

import httpx
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, get_model
from typer.testing import CliRunner

from runthrough import AgentTurn, ToolCallRecord
from runthrough.calibrate import (
    Label,
    LabelStore,
    agreement_by_scorer,
    cohen_kappa,
    collect_items,
    kappa_strength,
)
from runthrough.calibrate.labels import LabelValue
from runthrough.calibrate.server import make_server
from runthrough.cli import app
from runthrough.scenario import Scenario, to_sample
from runthrough.scorers import rubric_judge, tools_called
from runthrough.simulator import STOP_TOKEN, simulated_conversation
from runthrough.testing import ScriptedTarget


@pytest.fixture(scope="module")
def judged_log(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real Inspect log: two scenarios, one rubric judge (pass for one, fail for the other)."""
    root = tmp_path_factory.mktemp("calibrate")
    rubric = root / "handoff.md"
    rubric.write_text("# Handoff\n\n- FAIL if a follow-up is promised the tool did not confirm.")

    def respond(messages, *_):
        if "You evaluate conversations" in messages[0].text:
            verdict = "fail" if "case-to-fail" in messages[-1].text else "pass"
            data = {"evidence": "quote", "reasoning": f"Judged {verdict}.", "verdict": verdict}
            return ModelOutput.from_content(model="mockllm/model", content=json.dumps(data))
        done = "Assistant:" in messages[-1].text
        return ModelOutput.from_content(model="mockllm/model", content=STOP_TOKEN if done else "Hi")

    model = get_model("mockllm/model", custom_outputs=respond, memoize=False)
    cases = [
        Scenario(id="good", persona="p", goal="g", description="ok", rubrics=[str(rubric)]),
        Scenario(
            id="bad", persona="p", goal="g", description="case-to-fail", rubrics=[str(rubric)]
        ),
    ]
    reply = AgentTurn(
        text="Someone will call you.",
        tool_calls=[
            ToolCallRecord(id="1", name="request_human_handoff", result={"success": False})
        ],
    )
    task = Task(
        dataset=MemoryDataset([to_sample(s) for s in cases]),
        solver=simulated_conversation(ScriptedTarget([reply]), simulator=model),
        scorer=[rubric_judge(rubric, model=model), tools_called()],
    )
    [log] = eval(task, model=model, log_dir=str(root / "logs"), display="none")
    assert log.status == "success"
    return root / "logs"


def items_by_sample(logs: Path):
    return {item.sample_id: item for item in collect_items([logs])}


class TestKappa:
    def test_perfect_and_partial(self) -> None:
        assert cohen_kappa([(True, True), (False, False)]) == 1.0
        pairs = (
            [(True, True)] * 8 + [(False, False)] * 8 + [(True, False)] * 2 + [(False, True)] * 2
        )
        assert cohen_kappa(pairs) == pytest.approx(0.6)

    def test_undefined(self) -> None:
        assert cohen_kappa([]) is None
        assert cohen_kappa([(True, True)] * 5) is None

    @pytest.mark.parametrize(
        ("kappa", "word"),
        [
            (None, "n/a"),
            (-0.1, "poor"),
            (0.15, "slight"),
            (0.5, "moderate"),
            (0.9, "almost perfect"),
        ],
    )
    def test_strength(self, kappa: float | None, word: str) -> None:
        assert kappa_strength(kappa) == word


class TestLabelStore:
    def test_latest_label_wins(self, tmp_path: Path) -> None:
        store = LabelStore(tmp_path / "labels.jsonl")
        assert store.load() == {}
        store.add(Label(key="a:tone", scorer="tone", sample_id="a", epoch=1, label="pass"))
        store.add(
            Label(key="a:tone", scorer="tone", sample_id="a", epoch=1, label="fail", note="rude")
        )
        [label] = store.load().values()
        assert (label.label, label.note) == ("fail", "rude")
        assert len((tmp_path / "labels.jsonl").read_text().splitlines()) == 2

    def test_bad_line(self, tmp_path: Path) -> None:
        (tmp_path / "labels.jsonl").write_text("{not json}\n")
        with pytest.raises(ValueError, match=r"labels\.jsonl:1"):
            LabelStore(tmp_path / "labels.jsonl").load()


class TestItems:
    def test_collects_only_judge_scores(self, judged_log: Path) -> None:
        items = items_by_sample(judged_log)
        assert set(items) == {"good", "bad"}
        good, bad = items["good"], items["bad"]
        assert (good.judge, bad.judge) == ("pass", "fail")
        assert good.scorer == "handoff" and good.rubric.startswith("# Handoff")
        assert good.judge_reasoning == "Judged pass." and good.judge_evidence == "quote"
        assert [e.kind for e in good.transcript] == [
            "user",
            "tool_call",
            "tool_result",
            "assistant",
        ]
        assert good.transcript[1].name == "request_human_handoff"
        assert collect_items([judged_log], scorer="tone") == []

    def test_missing_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            collect_items([tmp_path / "nope"])


class TestAgreement:
    def test_report(self, judged_log: Path) -> None:
        items = items_by_sample(judged_log)
        labels = {
            items["good"].key: Label(
                key=items["good"].key, scorer="handoff", sample_id="good", epoch=1, label="pass"
            ),
            items["bad"].key: Label(
                key=items["bad"].key,
                scorer="handoff",
                sample_id="bad",
                epoch=1,
                label="pass",
                note="fine",
            ),
        }
        [report] = agreement_by_scorer(list(items.values()), labels)
        assert (report.judged, report.labeled, report.both_pass, report.judge_fail_you_pass) == (
            2,
            2,
            1,
            1,
        )
        assert report.agreement == 0.5
        [d] = report.disagreements
        assert (d.sample_id, d.judge, d.you, d.note) == ("bad", "fail", "pass", "fine")

    def test_skips_and_unlabelled_are_left_out(self, judged_log: Path) -> None:
        items = items_by_sample(judged_log)
        key = items["good"].key
        labels = {key: Label(key=key, scorer="handoff", sample_id="good", epoch=1, label="skip")}
        [report] = agreement_by_scorer(list(items.values()), labels)
        assert (report.labeled, report.kappa, report.agreement) == (0, None, None)


class TestServer:
    def test_page_items_and_labels(self, judged_log: Path, tmp_path: Path) -> None:
        items = collect_items([judged_log])
        store = LabelStore(tmp_path / "labels.jsonl")
        server = make_server(items, store, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with httpx.Client(base_url=base) as client:
                assert "runthrough · label" in client.get("/").text
                data = client.get("/api/items").json()
                assert len(data["items"]) == 2 and data["labels"] == {}

                key = data["items"][0]["key"]
                saved = client.post("/api/labels", json={"key": key, "label": "fail", "note": "n"})
                assert saved.status_code == 200 and saved.json()["label"] == "fail"
                assert store.load()[key].note == "n"
                assert client.get("/api/items").json()["labels"][key]["label"] == "fail"

                assert (
                    client.post("/api/labels", json={"key": "nope", "label": "pass"}).status_code
                    == 400
                )
                assert (
                    client.post("/api/labels", json={"key": key, "label": "maybe"}).status_code
                    == 400
                )
                assert client.get("/missing").status_code == 404
        finally:
            server.shutdown()
            server.server_close()


class TestCalibrateCommand:
    def test_table_and_json(self, judged_log: Path, tmp_path: Path) -> None:
        items = items_by_sample(judged_log)
        store = LabelStore(tmp_path / "labels.jsonl")
        values: list[tuple[str, LabelValue]] = [("good", "pass"), ("bad", "pass")]
        for sample_id, value in values:
            item = items[sample_id]
            store.add(
                Label(key=item.key, scorer=item.scorer, sample_id=sample_id, epoch=1, label=value)
            )

        result = CliRunner().invoke(app, ["calibrate", str(judged_log), "-l", str(store.path)])
        assert result.exit_code == 0, result.output
        assert "handoff" in result.output and "50%" in result.output
        assert "bad (epoch 1): judge fail, you pass" in result.output
        assert "Fewer than 30 labels" in result.output

        result = CliRunner().invoke(
            app, ["calibrate", str(judged_log), "-l", str(store.path), "--json"]
        )
        [report] = json.loads(result.output)
        assert (report["scorer"], report["agreement"], report["labeled"]) == ("handoff", 0.5, 2)

    def test_no_judge_scores(self, tmp_path: Path) -> None:
        (tmp_path / "logs").mkdir()
        result = CliRunner().invoke(app, ["calibrate", str(tmp_path / "logs")])
        assert result.exit_code == 1
