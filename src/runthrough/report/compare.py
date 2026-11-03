"""Compare runs of the same suite: pass rates, latency, tokens and cost per run.

A run is one or more Inspect logs with the same label. The label comes from the
task's metadata (``runthrough_label``), else the log's task arguments, else the
log file name.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.log import EvalLog, EvalSample, read_eval_log
from pydantic import BaseModel, Field

from runthrough.calibrate.items import find_logs

LABEL_KEY = "runthrough_label"


class Price(BaseModel):
    """USD per 1M tokens."""

    input: float
    output: float
    cached_input: float | None = None
    """Price for cached input tokens; defaults to the input price."""


class ScorerResult(BaseModel):
    name: str
    pass_rate: float | None
    stderr: float | None
    scored: int


class Usage(BaseModel):
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    turns_with_usage: int = 0


class RunSummary(BaseModel):
    label: str
    logs: list[str]
    models: list[str]
    """Models the agent reported in its reply metadata, if any."""
    conversations: int
    errors: int
    agent_turns: int
    turns_per_conversation: float | None
    latency_ms_mean: float | None
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    scorers: list[ScorerResult]
    overall_pass_rate: float | None
    """Mean of all applicable scorer verdicts across conversations."""
    usage: Usage
    tokens_per_conversation: float | None
    cost_usd: float | None
    cost_per_conversation_usd: float | None
    cost_per_1k_conversations_usd: float | None
    price: Price | None = None
    scenario_pass_rates: dict[str, float] = Field(default_factory=dict)
    """Scenario id -> share of applicable checks passed, over all epochs."""


def load_prices(path: str | Path) -> dict[str, Price]:
    """A YAML or JSON file mapping model id to ``{input, cached_input, output}`` (USD/1M)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {str(model): Price.model_validate(values) for model, values in data.items()}


def run_label(log: EvalLog, path: Path) -> str:
    metadata = log.eval.metadata or {}
    if metadata.get(LABEL_KEY):
        return str(metadata[LABEL_KEY])
    if log.eval.task_args:
        return ", ".join(f"{k}={v}" for k, v in sorted(log.eval.task_args.items()))
    return path.stem


def compare_runs(
    paths: Iterable[str | Path], *, prices: dict[str, Price] | None = None
) -> list[RunSummary]:
    """One summary per run label, sorted by label."""
    grouped: dict[str, list[tuple[Path, EvalLog]]] = defaultdict(list)
    for path in find_logs(paths):
        log = read_eval_log(path)
        if log.samples is None:
            continue
        grouped[run_label(log, path)].append((path, log))
    return [_summarise(label, grouped[label], prices or {}) for label in sorted(grouped)]


def _dedupe(runs: list[tuple[Path, EvalLog]]) -> list[EvalSample]:
    """One sample per (id, epoch) across a run's logs.

    A retried or resumed run leaves the first log in place and writes a new one that
    repeats the samples that already finished. Keep a sample without an error over one
    with an error, and otherwise the one from the newest log.
    """
    chosen: dict[tuple[str, int], tuple[tuple[bool, str], EvalSample]] = {}
    for _, log in runs:
        for sample in log.samples or []:
            key = (str(sample.id), sample.epoch)
            rank = (sample.error is None, log.eval.created)
            if key not in chosen or rank > chosen[key][0]:
                chosen[key] = (rank, sample)
    return [sample for _, sample in chosen.values()]


def _summarise(
    label: str, runs: list[tuple[Path, EvalLog]], prices: dict[str, Price]
) -> RunSummary:
    samples = _dedupe(runs)
    latencies: list[float] = []
    models: set[str] = set()
    usage = Usage()
    agent_turns = 0
    turns: list[int] = []
    verdicts: dict[str, list[float]] = defaultdict(list)
    scenario_verdicts: dict[str, list[float]] = defaultdict(list)

    for sample in samples:
        result = (sample.metadata or {}).get("runthrough", {})
        if "turns" in result:
            turns.append(int(result["turns"]))
        for message in sample.messages:
            if message.role != "assistant" or not message.metadata:
                continue
            meta = message.metadata
            if "latency_ms" in meta:
                latencies.append(float(meta["latency_ms"]))
                agent_turns += 1
            agent = meta.get("agent_metadata") or {}
            if agent.get("model"):
                models.add(str(agent["model"]))
            if isinstance(agent.get("usage"), dict):
                _add_usage(usage, agent["usage"])
        for name, score in (sample.scores or {}).items():
            value = _as_number(score.value)
            if value is not None:
                verdicts[name].append(value)
                scenario_verdicts[str(sample.id)].append(value)

    scorers = []
    for name in sorted(verdicts):
        values = verdicts[name]
        scorers.append(
            ScorerResult(
                name=name,
                pass_rate=statistics.fmean(values),
                stderr=(statistics.stdev(values) / len(values) ** 0.5) if len(values) > 1 else None,
                scored=len(values),
            )
        )
    all_values = [v for values in verdicts.values() for v in values]

    model = next(iter(models)) if len(models) == 1 else None
    price = prices.get(model) if model else prices.get(label)
    cost = _cost(usage, price) if price and usage.turns_with_usage else None
    conversations = len(samples)

    return RunSummary(
        label=label,
        logs=[path.name for path, _ in runs],
        models=sorted(models),
        conversations=conversations,
        errors=sum(1 for s in samples if s.error is not None),
        agent_turns=agent_turns,
        turns_per_conversation=statistics.fmean(turns) if turns else None,
        latency_ms_mean=statistics.fmean(latencies) if latencies else None,
        latency_ms_p50=_percentile(latencies, 50),
        latency_ms_p95=_percentile(latencies, 95),
        scorers=scorers,
        overall_pass_rate=statistics.fmean(all_values) if all_values else None,
        usage=usage,
        tokens_per_conversation=(
            (usage.input_tokens + usage.output_tokens) / conversations
            if usage.turns_with_usage and conversations
            else None
        ),
        cost_usd=cost,
        cost_per_conversation_usd=cost / conversations
        if cost is not None and conversations
        else None,
        cost_per_1k_conversations_usd=(
            cost / conversations * 1000 if cost is not None and conversations else None
        ),
        price=price,
        scenario_pass_rates={k: statistics.fmean(v) for k, v in sorted(scenario_verdicts.items())},
    )


def _add_usage(total: Usage, usage: dict[str, Any]) -> None:
    total.input_tokens += int(usage.get("inputTokens") or 0)
    total.cached_input_tokens += int(usage.get("cachedInputTokens") or 0)
    total.output_tokens += int(usage.get("outputTokens") or 0)
    total.reasoning_tokens += int(usage.get("reasoningTokens") or 0)
    total.turns_with_usage += 1


def _cost(usage: Usage, price: Price) -> float:
    """Output tokens already include reasoning tokens, as OpenAI reports them."""
    cached = usage.cached_input_tokens
    uncached = max(usage.input_tokens - cached, 0)
    cached_price = price.cached_input if price.cached_input is not None else price.input
    return (
        uncached * price.input + cached * cached_price + usage.output_tokens * price.output
    ) / 1e6


def _as_number(value: Any) -> float | None:
    if value in ("C", True):
        return 1.0
    if value in ("I", False, "N"):
        return 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


def _percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[pct - 1]
