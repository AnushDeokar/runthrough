"""LLM-as-a-judge scoring against rubric files.

Each rubric file becomes its own scorer, so ``inspect view`` shows one pass-rate
column per rubric. A rubric scorer only scores scenarios that list that rubric.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from inspect_ai.dataset import Dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, Model, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    NOANSWER,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState
from pydantic import BaseModel, ValidationError

from runthrough.scenario import Scenario, scenario_from_metadata
from runthrough.scorers._common import scenario_of
from runthrough.scorers.transcript import render_transcript

JUDGE_SYSTEM_PROMPT = """\
You evaluate conversations between a user and an AI assistant that a business has \
deployed. You get one rubric, some context about what the user wanted, and the full \
transcript. The transcript includes the assistant's tool calls and their results, \
which the user never saw.

Decide whether the assistant meets the rubric.
- Judge only what the rubric asks about.
- Base your verdict on what the transcript shows, including tool results. Do not assume \
a tool succeeded if its result says otherwise.
- If the rubric has several PASS/FAIL criteria, the verdict is "fail" if any FAIL \
criterion applies.

Reply with a single JSON object and nothing else:
{"evidence": "<short exact quote from the transcript that decides it, or empty>", \
"reasoning": "<one or two sentences>", "verdict": "pass" or "fail"}\
"""

JUDGE_USER_TEMPLATE = """\
# Rubric

{rubric}

# Context

What this scenario tests: {description}
What the user wanted: {goal}

# Transcript

{transcript}
"""

_RETRY_MESSAGE = (
    'Your reply was not valid. Reply with only the JSON object: {"evidence": "...", '
    '"reasoning": "...", "verdict": "pass" or "fail"}'
)


class Verdict(BaseModel):
    evidence: str = ""
    reasoning: str = ""
    verdict: Literal["pass", "fail"]


def rubric_scorers(
    dataset: Dataset | Iterable[Scenario],
    *,
    model: str | Model | None = None,
    max_tool_chars: int = 8000,
) -> list[Scorer]:
    """One judge scorer per rubric file used by the scenarios in ``dataset``.

    Args:
        dataset: The dataset from :func:`runthrough.scenarios`, or scenarios.
        model: Judge model. Defaults to the ``grader`` model role, then the main model.
        max_tool_chars: Cut tool results longer than this in the judge's transcript.
    """
    rubrics: dict[str, str] = {}
    for scenario in _scenarios(dataset):
        for path in scenario.rubrics:
            name = rubric_name(path)
            if rubrics.setdefault(name, path) != path:
                raise ValueError(
                    f"Two rubric files share the name '{name}': {rubrics[name]} and {path}. "
                    "Rename one of them."
                )
    return [
        rubric_judge(path, name=name, model=model, max_tool_chars=max_tool_chars)
        for name, path in sorted(rubrics.items())
    ]


def rubric_name(path: str | Path) -> str:
    """Scorer name for a rubric file: its file name, lowercase, with ``_`` separators."""
    return re.sub(r"[^a-z0-9]+", "_", Path(path).stem.lower()).strip("_")


def rubric_judge(
    rubric: str | Path,
    *,
    name: str | None = None,
    model: str | Model | None = None,
    max_tool_chars: int = 8000,
) -> Scorer:
    """A scorer that judges conversations against one rubric file.

    Only scenarios that list this rubric are scored; others return ``None``.
    """
    rubric_path = Path(rubric).resolve()
    rubric_text = rubric_path.read_text(encoding="utf-8").strip()

    @scorer(metrics=[accuracy(), stderr()], name=name or rubric_name(rubric_path))
    def judge() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score | None:
            scenario = scenario_of(state)
            if str(rubric_path) not in scenario.rubrics:
                return None
            grader = get_model(model, role="grader")
            prompt = JUDGE_USER_TEMPLATE.format(
                rubric=rubric_text,
                description=scenario.description or "(not given)",
                goal=scenario.goal,
                transcript=render_transcript(state.messages, max_tool_chars=max_tool_chars),
            )
            messages = [
                ChatMessageSystem(content=JUDGE_SYSTEM_PROMPT),
                ChatMessageUser(content=prompt),
            ]

            output = await grader.generate(messages)
            verdict = parse_verdict(output.completion)
            if verdict is None:
                messages += [output.message, ChatMessageUser(content=_RETRY_MESSAGE)]
                output = await grader.generate(messages)
                verdict = parse_verdict(output.completion)
            if verdict is None:
                return Score(
                    value=NOANSWER,
                    explanation=f"The judge's reply could not be read: {output.completion[:500]}",
                    metadata={"rubric": str(rubric_path), "judge": str(grader)},
                )

            explanation = f"{verdict.verdict.upper()}: {verdict.reasoning}"
            if verdict.evidence:
                explanation += f'\nEvidence: "{verdict.evidence}"'
            return Score(
                value=CORRECT if verdict.verdict == "pass" else INCORRECT,
                answer=verdict.verdict,
                explanation=explanation,
                metadata={
                    "rubric": str(rubric_path),
                    "judge": str(grader),
                    "evidence": verdict.evidence,
                    "reasoning": verdict.reasoning,
                },
            )

        return score

    return judge()


def parse_verdict(text: str) -> Verdict | None:
    """Read the judge's JSON verdict, tolerating code fences and text around it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and isinstance(data.get("verdict"), str):
        data["verdict"] = data["verdict"].strip().lower()
    try:
        return Verdict.model_validate(data)
    except ValidationError:
        return None


def _scenarios(dataset: Dataset | Iterable[Scenario]) -> list[Scenario]:
    items = list(dataset)
    if items and isinstance(items[0], Scenario):
        return items  # type: ignore[return-value]
    return [scenario_from_metadata(sample.metadata or {}) for sample in items]  # type: ignore[union-attr]
