"""The scenario file format.

A scenario describes one simulated user: who they are (``persona``), what they want
(``goal``), and what the agent is expected to do. One YAML file holds one scenario or a
list of scenarios.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"


class Expectations(BaseModel):
    """Checks on the agent's behaviour that need no LLM judge."""

    model_config = ConfigDict(extra="forbid")

    tools_called: list[str] = Field(default_factory=list)
    """Tools the agent must call at least once."""

    tools_not_called: list[str] = Field(default_factory=list)
    """Tools the agent must never call."""

    tool_order: list[str] = Field(default_factory=list)
    """Tools that must be called in this relative order. Other calls may come in between."""

    language: str | None = None
    """ISO 639-1 code every agent reply must be written in, e.g. ``en`` or ``ar``."""

    @model_validator(mode="after")
    def _no_contradictions(self) -> Expectations:
        forbidden = set(self.tools_not_called)
        for field in ("tools_called", "tool_order"):
            clash = sorted(forbidden & set(getattr(self, field)))
            if clash:
                raise ValueError(f"{', '.join(clash)} is in both {field} and tools_not_called")
        return self


class Scenario(BaseModel):
    """One simulated conversation to run against the agent."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(pattern=ID_PATTERN)
    """Unique within a suite. Letters, digits, ``_``, ``-`` and ``.``."""

    persona: str = Field(min_length=1)
    """Who the simulated user is and how they behave."""

    goal: str = Field(min_length=1)
    """What the simulated user is trying to get done."""

    description: str | None = None
    """What this scenario tests, for people reading the suite."""

    opening: list[str] = Field(default_factory=list)
    """Exact user messages to send first, before the simulator takes over."""

    max_turns: int = Field(default=8, ge=1, le=50)
    """Maximum number of user messages in the conversation."""

    context: dict[str, Any] = Field(default_factory=dict)
    """Values passed to the target adapter on every turn, e.g. profile data or a prompt key."""

    expect: Expectations = Field(default_factory=Expectations)

    rubrics: list[str] = Field(default_factory=list)
    """Rubric files for the LLM judge, relative to the scenario file."""

    tags: list[str] = Field(default_factory=list)

    source_file: str | None = None
    """Set by the loader to the file the scenario came from."""

    @field_validator("opening", mode="before")
    @classmethod
    def _single_opening(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value

    @model_validator(mode="after")
    def _opening_fits(self) -> Scenario:
        if len(self.opening) > self.max_turns:
            raise ValueError(
                f"opening has {len(self.opening)} messages but max_turns is {self.max_turns}"
            )
        return self
