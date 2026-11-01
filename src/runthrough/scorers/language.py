"""Check that the agent replies in the language the scenario expects."""

from __future__ import annotations

import re
from collections.abc import Callable

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState

from runthrough.scorers._common import agent_replies, scenario_of

LanguageDetector = Callable[[str], str | None]
"""Returns an ISO 639-1 code for a text, or ``None`` if unsure."""

_NOISE = re.compile(
    r"\[([^\]]*)\]\([^)]*\)"  # markdown links: keep the label
    r"|https?://\S+|www\.\S+"  # bare URLs
    r"|\S+@\S+\.\S+"  # emails
    r"|[+\d][\d\s().-]{6,}\d"  # phone numbers
)


@scorer(metrics=[accuracy(), stderr()])
def reply_language(detector: LanguageDetector | None = None, min_letters: int = 12) -> Scorer:
    """Every agent reply is written in ``expect.language``.

    Links, emails and phone numbers are ignored, and replies with fewer than
    ``min_letters`` letters are skipped as too short to judge. Not scored when the
    scenario sets no language, or no reply is long enough.

    Args:
        detector: Function returning a language code for a text. Defaults to
            ``langdetect`` (``pip install 'runthrough[language]'``).
        min_letters: Skip replies shorter than this.
    """

    async def score(state: TaskState, target: Target) -> Score | None:
        expected = scenario_of(state).expect.language
        if expected is None:
            return None
        detect = detector or _langdetect
        checked: list[tuple[int, str]] = []
        wrong: list[str] = []
        for turn, text in agent_replies(state.messages):
            cleaned = _NOISE.sub(lambda m: m.group(1) or " ", text)
            if sum(ch.isalpha() for ch in cleaned) < min_letters:
                continue
            found = detect(cleaned)
            checked.append((turn, found or "unknown"))
            if found != expected.lower():
                wrong.append(f"turn {turn}: {found or 'unknown'}")
        if not checked:
            return None
        return Score(
            value=INCORRECT if wrong else CORRECT,
            answer=", ".join(sorted({lang for _, lang in checked})),
            explanation=(
                f"Expected {expected}; " + "; ".join(wrong)
                if wrong
                else f"All {len(checked)} checked replies are in {expected}."
            ),
            metadata={"expected": expected, "detected": [lang for _, lang in checked]},
        )

    return score


def _langdetect(text: str) -> str | None:
    try:
        from langdetect import DetectorFactory, LangDetectException, detect
    except ImportError as exc:
        raise ImportError(
            "reply_language needs langdetect. Install it with: pip install 'runthrough[language]'"
        ) from exc
    DetectorFactory.seed = 0  # deterministic results
    try:
        return detect(text)
    except LangDetectException:
        return None
