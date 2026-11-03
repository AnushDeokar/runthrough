"""How well an LLM judge agrees with your labels."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel

from runthrough.calibrate.items import JudgedItem
from runthrough.calibrate.labels import Label


class Disagreement(BaseModel):
    sample_id: str
    epoch: int
    judge: str
    you: str
    judge_reasoning: str
    note: str
    log: str


class Agreement(BaseModel):
    """Judge vs. your labels for one rubric scorer."""

    scorer: str
    judged: int
    """Conversations the judge scored with a readable verdict."""
    labeled: int
    """Of those, how many you labelled pass or fail."""
    both_pass: int
    both_fail: int
    judge_pass_you_fail: int
    judge_fail_you_pass: int
    unreadable: int
    kappa: float | None
    disagreements: list[Disagreement]

    @property
    def agreement(self) -> float | None:
        if self.labeled == 0:
            return None
        return (self.both_pass + self.both_fail) / self.labeled

    @property
    def strength(self) -> str:
        """Landis & Koch's wording for the kappa value."""
        return kappa_strength(self.kappa)


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa for two raters' pass/fail decisions.

    ``None`` when it is undefined: no pairs, or both raters always gave the same single
    answer (chance agreement is 1).
    """
    n = len(pairs)
    if n == 0:
        return None
    observed = sum(a == b for a, b in pairs) / n
    a_pass = sum(a for a, _ in pairs) / n
    b_pass = sum(b for _, b in pairs) / n
    expected = a_pass * b_pass + (1 - a_pass) * (1 - b_pass)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def kappa_strength(kappa: float | None) -> str:
    if kappa is None:
        return "n/a"
    for bound, word in (
        (0, "poor"),
        (0.2, "slight"),
        (0.4, "fair"),
        (0.6, "moderate"),
        (0.8, "substantial"),
    ):
        if kappa <= bound:
            return word
    return "almost perfect"


def agreement_by_scorer(items: list[JudgedItem], labels: dict[str, Label]) -> list[Agreement]:
    """One :class:`Agreement` per rubric scorer that appears in ``items``."""
    grouped: dict[str, list[JudgedItem]] = defaultdict(list)
    for item in items:
        grouped[item.scorer].append(item)

    results = []
    for scorer, scorer_items in sorted(grouped.items()):
        counts = {
            "both_pass": 0,
            "both_fail": 0,
            "judge_pass_you_fail": 0,
            "judge_fail_you_pass": 0,
        }
        pairs: list[tuple[bool, bool]] = []
        disagreements: list[Disagreement] = []
        unreadable = 0
        for item in scorer_items:
            if item.judge == "unreadable":
                unreadable += 1
                continue
            label = labels.get(item.key)
            if label is None or label.label == "skip":
                continue
            judge_pass, you_pass = item.judge == "pass", label.label == "pass"
            pairs.append((judge_pass, you_pass))
            if judge_pass and you_pass:
                counts["both_pass"] += 1
            elif not judge_pass and not you_pass:
                counts["both_fail"] += 1
            else:
                counts["judge_pass_you_fail" if judge_pass else "judge_fail_you_pass"] += 1
                disagreements.append(
                    Disagreement(
                        sample_id=item.sample_id,
                        epoch=item.epoch,
                        judge=item.judge,
                        you=label.label,
                        judge_reasoning=item.judge_reasoning,
                        note=label.note,
                        log=item.log,
                    )
                )
        results.append(
            Agreement(
                scorer=scorer,
                judged=len(scorer_items) - unreadable,
                labeled=len(pairs),
                unreadable=unreadable,
                kappa=cohen_kappa(pairs),
                disagreements=disagreements,
                **counts,
            )
        )
    return results
