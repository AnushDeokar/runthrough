"""Load scenario files and turn them into an Inspect dataset."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml
from inspect_ai.dataset import MemoryDataset, Sample
from pydantic import ValidationError

from runthrough.scenario.schema import Scenario

SCENARIO_KEY = "runthrough_scenario"
"""Key under which the full scenario is stored in each Inspect sample's metadata."""

_SUFFIXES = (".yaml", ".yml")


class ScenarioError(ValueError):
    """One or more scenario files are invalid. ``problems`` lists every issue found."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        noun = "problem" if len(problems) == 1 else "problems"
        super().__init__(f"{len(problems)} scenario {noun}:\n" + "\n".join(problems))


def load_scenarios(paths: str | Path | Iterable[str | Path]) -> list[Scenario]:
    """Load scenarios from YAML files and/or directories (searched recursively).

    Every file is checked before anything is returned, so all problems are reported
    together in one :class:`ScenarioError`.
    """
    if isinstance(paths, str | Path):
        paths = [paths]

    problems: list[str] = []
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(p for p in path.rglob("*") if p.suffix in _SUFFIXES and p.is_file())
            if not found:
                problems.append(f"{path}: no .yaml or .yml files found")
            files.extend(found)
        elif path.is_file():
            files.append(path)
        else:
            problems.append(f"{path}: file or directory not found")

    scenarios: list[Scenario] = []
    for file in files:
        scenarios.extend(_load_file(file, problems))

    seen: dict[str, str] = {}
    for scenario in scenarios:
        if scenario.id in seen:
            problems.append(
                f"{scenario.source_file}: duplicate id '{scenario.id}' "
                f"(first defined in {seen[scenario.id]})"
            )
        else:
            seen[scenario.id] = scenario.source_file or "?"

    if problems:
        raise ScenarioError(problems)
    return scenarios


def scenarios(
    paths: str | Path | Iterable[str | Path],
    *,
    tags: Iterable[str] | None = None,
    name: str | None = None,
) -> MemoryDataset:
    """Load scenarios as an Inspect dataset, one sample per scenario.

    Args:
        paths: Scenario files and/or directories.
        tags: If given, keep only scenarios that have at least one of these tags.
        name: Dataset name shown in Inspect logs. Defaults to the first path's name.
    """
    if isinstance(paths, str | Path):
        paths = [paths]
    paths = list(paths)
    loaded = load_scenarios(paths)
    if tags is not None:
        wanted = set(tags)
        loaded = [s for s in loaded if wanted & set(s.tags)]
    return MemoryDataset(
        [to_sample(s) for s in loaded],
        name=name or (Path(paths[0]).stem if paths else None),
    )


def to_sample(scenario: Scenario) -> Sample:
    """Convert a scenario to an Inspect sample. The scenario is kept whole in metadata."""
    return Sample(
        id=scenario.id,
        input=f"Persona: {scenario.persona}\nGoal: {scenario.goal}",
        metadata={SCENARIO_KEY: scenario.model_dump(mode="json")},
    )


def scenario_from_metadata(metadata: Mapping[str, Any]) -> Scenario:
    """Recover the scenario from a sample's (or TaskState's) metadata."""
    try:
        data = metadata[SCENARIO_KEY]
    except KeyError:
        raise ValueError(
            "This sample has no runthrough scenario. Load the dataset with runthrough.scenarios()."
        ) from None
    return Scenario.model_validate(data)


def _load_file(file: Path, problems: list[str]) -> list[Scenario]:
    try:
        data = yaml.safe_load(file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        problems.append(f"{file}: invalid YAML: {exc}")
        return []

    if data is None:
        problems.append(f"{file}: file is empty")
        return []
    entries = data if isinstance(data, list) else [data]
    many = isinstance(data, list)

    loaded: list[Scenario] = []
    for index, entry in enumerate(entries):
        where = f"{file}[{index}]" if many else str(file)
        if not isinstance(entry, dict):
            problems.append(f"{where}: expected a mapping of fields, got {type(entry).__name__}")
            continue
        if "source_file" in entry:
            problems.append(f"{where}: source_file is set by the loader; remove it")
            continue
        try:
            scenario = Scenario.model_validate({**entry, "source_file": str(file)})
        except ValidationError as exc:
            problems.extend(_format_errors(where, exc))
            continue

        resolved, missing = _resolve_rubrics(file.parent, scenario.rubrics)
        problems.extend(f"{where}: rubric file not found: {m}" for m in missing)
        if missing:
            continue
        loaded.append(scenario.model_copy(update={"rubrics": resolved}))
    return loaded


def _resolve_rubrics(base: Path, rubrics: list[str]) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    missing: list[str] = []
    for rubric in rubrics:
        path = (base / rubric).resolve()
        if path.is_file():
            resolved.append(str(path))
        else:
            missing.append(rubric)
    return resolved, missing


def _format_errors(where: str, exc: ValidationError) -> list[str]:
    lines = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error["loc"]) or "(scenario)"
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"{where}: {loc}: {message}")
    return lines
