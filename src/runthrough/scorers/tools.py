"""Exact checks on the tools the agent called. No LLM involved.

Each scorer reads the scenario's ``expect`` block and returns ``None`` (not scored)
for scenarios that don't set the expectation it checks, so pass rates only count
scenarios where the check applies.
"""

from __future__ import annotations

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from runthrough.scorers._common import called_tools, is_subsequence, join, scenario_of
from runthrough.simulator.solver import RESULT_KEY


@scorer(metrics=[accuracy(), stderr()])
def tools_called() -> Scorer:
    """Every tool in ``expect.tools_called`` was called at least once."""

    async def score(state: TaskState, target: Target) -> Score | None:
        expected = scenario_of(state).expect.tools_called
        if not expected:
            return None
        called = [c.name for c in called_tools(state.messages)]
        missing = [name for name in expected if name not in called]
        return Score(
            value=INCORRECT if missing else CORRECT,
            answer=join(called),
            explanation=(
                f"Missing: {join(missing)}. Called: {join(called)}."
                if missing
                else f"All expected tools were called ({join(expected)})."
            ),
            metadata={"expected": expected, "called": called, "missing": missing},
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def tools_not_called() -> Scorer:
    """No tool in ``expect.tools_not_called`` was called."""

    async def score(state: TaskState, target: Target) -> Score | None:
        forbidden = scenario_of(state).expect.tools_not_called
        if not forbidden:
            return None
        calls = called_tools(state.messages)
        offending = [c for c in calls if c.name in forbidden]
        return Score(
            value=INCORRECT if offending else CORRECT,
            answer=join([c.name for c in calls]),
            explanation=(
                "Called forbidden tools: "
                + ", ".join(f"{c.name} (turn {c.turn})" for c in offending)
                if offending
                else f"None of {join(forbidden)} was called."
            ),
            metadata={"forbidden": forbidden, "offending": [c.name for c in offending]},
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def tool_order() -> Scorer:
    """The tools in ``expect.tool_order`` were called in that relative order."""

    async def score(state: TaskState, target: Target) -> Score | None:
        expected = scenario_of(state).expect.tool_order
        if not expected:
            return None
        called = [c.name for c in called_tools(state.messages)]
        ok = is_subsequence(expected, called)
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=" > ".join(called) or "none",
            explanation=(
                f"Expected order {' > '.join(expected)}; actual calls: "
                f"{' > '.join(called) or 'none'}."
            ),
            metadata={"expected": expected, "called": called},
        )

    return score


@scorer(metrics=[accuracy(), stderr()])
def no_agent_errors() -> Scorer:
    """The agent answered every turn without an error (HTTP 5xx or a stream error)."""

    async def score(state: TaskState, target: Target) -> Score:
        result = state.metadata.get(RESULT_KEY, {})
        errors = [
            str(m.metadata["error"])
            for m in state.messages
            if m.role == "assistant" and m.metadata and "error" in m.metadata
        ]
        failed = bool(errors) or result.get("stop_reason") == "agent_error"
        return Score(
            value=INCORRECT if failed else CORRECT,
            explanation="; ".join(errors) if errors else "No agent errors.",
        )

    return score
