import json
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.model import ModelOutput, get_model
from typer.testing import CliRunner

from runthrough import AgentTurn, ToolCallRecord
from runthrough.adapters.ai_sdk import parse_ui_message_stream
from runthrough.cli import app
from runthrough.report import Price, compare_runs, load_prices
from runthrough.scenario import Scenario, to_sample
from runthrough.scorers import tools_called
from runthrough.simulator import STOP_TOKEN, simulated_conversation
from runthrough.testing import ScriptedTarget

USAGE = {"inputTokens": 1000, "cachedInputTokens": 400, "outputTokens": 200, "reasoningTokens": 50}


def run(label: str, reply: AgentTurn, root: Path) -> None:
    sim = get_model(
        "mockllm/model",
        custom_outputs=lambda m, *_: ModelOutput.from_content(
            model="mockllm/model", content=STOP_TOKEN if "Assistant:" in m[-1].text else "Hi"
        ),
        memoize=False,
    )
    cases = [
        Scenario.model_validate(
            {"id": f"s{i}", "persona": "p", "goal": "g", "expect": {"tools_called": ["lookup"]}}
        )
        for i in (1, 2)
    ]
    task = Task(
        dataset=MemoryDataset([to_sample(s) for s in cases]),
        solver=simulated_conversation(ScriptedTarget([reply]), simulator=sim),
        scorer=[tools_called()],
        metadata={"runthrough_label": label},
    )
    [log] = eval(task, model=sim, log_dir=str(root), display="none")
    assert log.status == "success"


@pytest.fixture(scope="module")
def logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("compare")
    good = AgentTurn(
        text="Done.",
        latency_ms=800,
        tool_calls=[ToolCallRecord(id="1", name="lookup")],
        metadata={"model": "model-a", "usage": USAGE},
    )
    bad = AgentTurn(text="No.", latency_ms=1200, metadata={"model": "model-b", "usage": USAGE})
    run("model-a", good, root)
    run("model-b", bad, root)
    return root


def test_adapter_keeps_message_metadata() -> None:
    turn = parse_ui_message_stream(
        [
            {"type": "start", "messageId": "m1"},
            {"type": "text-delta", "id": "t", "delta": "Hi"},
            {"type": "finish", "messageMetadata": {"model": "gpt-x", "usage": USAGE}},
        ]
    )
    assert turn.metadata == {"model": "gpt-x", "usage": USAGE}


def test_compare_runs(logs: Path) -> None:
    prices = {"model-a": Price(input=1.0, cached_input=0.1, output=10.0)}
    a, b = compare_runs([logs], prices=prices)
    assert (a.label, a.models, a.conversations, a.agent_turns) == ("model-a", ["model-a"], 2, 2)
    assert a.scorers[0].pass_rate == 1.0 and b.scorers[0].pass_rate == 0.0
    assert a.scenario_pass_rates == {"s1": 1.0, "s2": 1.0}
    assert a.latency_ms_p50 == 800 and b.latency_ms_mean == 1200
    assert a.usage.input_tokens == 2000 and a.usage.cached_input_tokens == 800
    # per turn: 600 uncached * $1 + 400 cached * $0.1 + 200 out * $10, per 1M tokens
    assert a.cost_usd == pytest.approx(2 * (600 * 1 + 400 * 0.1 + 200 * 10) / 1e6)
    assert a.tokens_per_conversation == 1200
    assert b.cost_usd is None  # no price for model-b


def test_load_prices(tmp_path: Path) -> None:
    path = tmp_path / "prices.yaml"
    path.write_text("gpt-x: {input: 0.5, output: 2}\n")
    assert load_prices(path) == {"gpt-x": Price(input=0.5, output=2)}


def test_compare_command(logs: Path, tmp_path: Path) -> None:
    prices = tmp_path / "prices.yaml"
    prices.write_text("model-a: {input: 1, cached_input: 0.1, output: 10}\n")
    result = CliRunner().invoke(app, ["compare", str(logs), "--prices", str(prices)])
    assert result.exit_code == 0, result.output
    assert "model-a" in result.output and "Pass rate by scorer" in result.output
    data = json.loads(CliRunner().invoke(app, ["compare", str(logs), "--json"]).output)
    assert [r["label"] for r in data] == ["model-a", "model-b"]


def test_retried_samples_are_counted_once(logs: Path, tmp_path: Path) -> None:
    import shutil

    copy = tmp_path / "copy"
    shutil.copytree(logs, copy)
    for path in list(copy.glob("*.eval")):
        shutil.copy(path, path.with_name("retry-" + path.name))
    a, _ = compare_runs([copy])
    assert len(a.logs) == 2 and a.conversations == 2 and a.agent_turns == 2
