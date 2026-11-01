"""Example suite wired to a scripted stand-in agent.

Run it with any simulator model, e.g. a local one:

    inspect eval examples/suites/support/task.py --model ollama/qwen3:8b
    inspect view

Swap ``agent`` for a real adapter to test your own endpoint.
"""

from pathlib import Path

from inspect_ai import Task, task

from runthrough import AgentTurn, ToolCallRecord, scenarios, simulated_conversation
from runthrough.testing import ScriptedTarget

HERE = Path(__file__).parent

agent = ScriptedTarget(
    [
        "Sorry to hear that! Could you give me your order number?",
        AgentTurn(
            text="I've found order 48213 and issued a full refund. It should arrive in 3-5 days.",
            tool_calls=[
                ToolCallRecord(id="1", name="lookup_order", args={"order_id": "48213"}),
                ToolCallRecord(id="2", name="issue_refund", args={"order_id": "48213"}),
            ],
        ),
        "Is there anything else I can help with?",
    ]
)


@task
def support() -> Task:
    return Task(
        dataset=scenarios(HERE / "scenarios"),
        solver=simulated_conversation(agent),
    )
