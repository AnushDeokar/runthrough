"""Collect the conversations an LLM judge scored, from Inspect ``.eval`` logs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.scorer import CORRECT, INCORRECT
from pydantic import BaseModel

JudgeVerdict = Literal["pass", "fail", "unreadable"]


class TranscriptEntry(BaseModel):
    kind: Literal["user", "assistant", "tool_call", "tool_result", "error"]
    text: str
    name: str | None = None


class JudgedItem(BaseModel):
    """One conversation scored by one rubric judge."""

    key: str
    """``<sample uuid>:<scorer name>``; what a label is attached to."""
    conversation: str
    """``<sample uuid>``; shared by every rubric judged on the same conversation."""
    scorer: str
    sample_id: str
    epoch: int
    log: str
    rubric_path: str
    rubric: str
    judge: JudgeVerdict
    judge_reasoning: str = ""
    judge_evidence: str = ""
    goal: str = ""
    description: str = ""
    transcript: list[TranscriptEntry]


class RubricVerdict(BaseModel):
    """One rubric judge's verdict on a conversation (what you label)."""

    key: str
    scorer: str
    rubric_path: str
    rubric: str
    judge: JudgeVerdict
    judge_reasoning: str = ""
    judge_evidence: str = ""


class JudgedConversation(BaseModel):
    """A conversation with every rubric verdict the judge gave it."""

    key: str
    sample_id: str
    epoch: int
    log: str
    goal: str = ""
    description: str = ""
    transcript: list[TranscriptEntry]
    verdicts: list[RubricVerdict]


def group_by_conversation(items: Iterable[JudgedItem]) -> list[JudgedConversation]:
    """Group per-rubric items so each conversation appears once, in first-seen order."""
    groups: dict[str, JudgedConversation] = {}
    for item in items:
        group = groups.get(item.conversation)
        if group is None:
            group = groups[item.conversation] = JudgedConversation(
                key=item.conversation,
                sample_id=item.sample_id,
                epoch=item.epoch,
                log=item.log,
                goal=item.goal,
                description=item.description,
                transcript=item.transcript,
                verdicts=[],
            )
        group.verdicts.append(
            RubricVerdict(
                key=item.key,
                scorer=item.scorer,
                rubric_path=item.rubric_path,
                rubric=item.rubric,
                judge=item.judge,
                judge_reasoning=item.judge_reasoning,
                judge_evidence=item.judge_evidence,
            )
        )
    for group in groups.values():
        group.verdicts.sort(key=lambda v: v.scorer)
    return list(groups.values())


def find_logs(paths: Iterable[str | Path]) -> list[Path]:
    """``.eval`` and ``.json`` logs in the given files and directories (recursive)."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(sorted(p for p in path.rglob("*") if p.suffix in (".eval", ".json")))
        elif path.is_file():
            found.append(path)
        else:
            raise FileNotFoundError(f"{path}: file or directory not found")
    return found


def collect_items(paths: Iterable[str | Path], *, scorer: str | None = None) -> list[JudgedItem]:
    """Every rubric-judge score in the logs, optionally for one scorer only."""
    items: list[JudgedItem] = []
    for log_path in find_logs(paths):
        log = read_eval_log(log_path, resolve_attachments=True)
        for sample in log.samples or []:
            items.extend(_items_for(sample, log_path, scorer))
    return items


def _items_for(sample: EvalSample, log_path: Path, only: str | None) -> list[JudgedItem]:
    items = []
    scenario = (sample.metadata or {}).get("runthrough_scenario", {})
    for name, score in (sample.scores or {}).items():
        metadata = score.metadata or {}
        if "rubric" not in metadata or "judge" not in metadata:
            continue  # not a rubric judge
        if only is not None and name != only:
            continue
        rubric_path = str(metadata["rubric"])
        items.append(
            JudgedItem(
                key=f"{sample.uuid}:{name}",
                conversation=str(sample.uuid),
                scorer=name,
                sample_id=str(sample.id),
                epoch=sample.epoch,
                log=log_path.name,
                rubric_path=rubric_path,
                rubric=_read(rubric_path),
                judge=_verdict(score.value),
                judge_reasoning=str(metadata.get("reasoning", score.explanation or "")),
                judge_evidence=str(metadata.get("evidence", "")),
                goal=str(scenario.get("goal", "")),
                description=str(scenario.get("description") or ""),
                transcript=_transcript(sample),
            )
        )
    return items


def _verdict(value: Any) -> JudgeVerdict:
    if value == CORRECT:
        return "pass"
    if value == INCORRECT:
        return "fail"
    return "unreadable"


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return f"(rubric file not found: {path})"


def _transcript(sample: EvalSample) -> list[TranscriptEntry]:
    entries: list[TranscriptEntry] = []
    for message in sample.messages:
        if isinstance(message, ChatMessageUser):
            entries.append(TranscriptEntry(kind="user", text=message.text))
        elif isinstance(message, ChatMessageAssistant):
            if message.text.strip():
                entries.append(TranscriptEntry(kind="assistant", text=message.text))
            for call in message.tool_calls or []:
                entries.append(
                    TranscriptEntry(
                        kind="tool_call",
                        name=call.function,
                        text=json.dumps(call.arguments, ensure_ascii=False, indent=2),
                    )
                )
            if message.metadata and "error" in message.metadata:
                entries.append(TranscriptEntry(kind="error", text=str(message.metadata["error"])))
        elif isinstance(message, ChatMessageTool):
            text = message.error.message if message.error is not None else message.text
            entries.append(
                TranscriptEntry(
                    kind="error" if message.error is not None else "tool_result",
                    name=message.function,
                    text=text,
                )
            )
    return entries
