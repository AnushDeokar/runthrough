"""runthrough: evaluate deployed chat agents with simulated users, on Inspect AI."""

from importlib.metadata import PackageNotFoundError, version

from runthrough.adapters import (
    AgentStep,
    AgentTurn,
    Target,
    ToolCallRecord,
    TurnContext,
    ai_sdk_endpoint,
)
from runthrough.scenario import (
    Expectations,
    Scenario,
    ScenarioError,
    load_scenarios,
    scenarios,
)
from runthrough.simulator import simulated_conversation

try:
    __version__ = version("runthrough")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

__all__ = [
    "AgentStep",
    "AgentTurn",
    "Expectations",
    "Scenario",
    "ScenarioError",
    "Target",
    "ToolCallRecord",
    "TurnContext",
    "__version__",
    "ai_sdk_endpoint",
    "load_scenarios",
    "scenarios",
    "simulated_conversation",
]
