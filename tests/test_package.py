from importlib.metadata import entry_points
from typing import Literal

from typer.testing import CliRunner

import runthrough
from runthrough import AgentTurn, Target, ToolCallRecord, TurnContext
from runthrough.cli import app


def test_version_is_set() -> None:
    assert runthrough.__version__ != "0.0.0"


def test_inspect_entry_point_is_registered() -> None:
    eps = entry_points(group="inspect_ai")
    ep = next(ep for ep in eps if ep.name == "runthrough")
    assert ep.value == "runthrough._registry"
    ep.load()


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"runthrough {runthrough.__version__}"


def test_agent_turn_defaults() -> None:
    turn = AgentTurn(
        text="I've passed your request to admissions.",
        tool_calls=[ToolCallRecord(id="call_1", name="request_human_handoff")],
    )
    assert turn.tool_calls[0].args == {}
    assert turn.error is None


class _EchoTarget:
    history: Literal["full", "session"] = "full"

    async def send(self, messages, ctx: TurnContext) -> AgentTurn:
        return AgentTurn(text=f"turn {ctx.turn}: {len(messages)} messages")


async def test_target_protocol() -> None:
    target = _EchoTarget()
    assert isinstance(target, Target)
    turn = await target.send([], TurnContext(conversation_id="c1", turn=1))
    assert turn.text == "turn 1: 0 messages"
