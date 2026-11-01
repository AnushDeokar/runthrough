"""Scorers for conversation trajectories: exact checks and the rubric judge."""

from __future__ import annotations

from collections.abc import Iterable

from inspect_ai.dataset import Dataset
from inspect_ai.model import Model
from inspect_ai.scorer import Scorer

from runthrough.scenario import Scenario
from runthrough.scorers.judge import parse_verdict, rubric_judge, rubric_name, rubric_scorers
from runthrough.scorers.language import reply_language
from runthrough.scorers.tools import no_agent_errors, tool_order, tools_called, tools_not_called
from runthrough.scorers.transcript import render_transcript


def expectation_scorers() -> list[Scorer]:
    """Exact checks for every ``expect`` field, plus agent errors. No LLM needed."""
    return [tools_called(), tools_not_called(), tool_order(), reply_language(), no_agent_errors()]


def scenario_scorers(
    dataset: Dataset | Iterable[Scenario], *, judge: str | Model | None = None
) -> list[Scorer]:
    """Everything a suite's scenarios ask for: exact checks and one judge per rubric.

    Args:
        dataset: The dataset from :func:`runthrough.scenarios`.
        judge: Judge model. Defaults to the ``grader`` model role, then the main model.
    """
    return [*expectation_scorers(), *rubric_scorers(dataset, model=judge)]


__all__ = [
    "expectation_scorers",
    "no_agent_errors",
    "parse_verdict",
    "render_transcript",
    "reply_language",
    "rubric_judge",
    "rubric_name",
    "rubric_scorers",
    "scenario_scorers",
    "tool_order",
    "tools_called",
    "tools_not_called",
]
