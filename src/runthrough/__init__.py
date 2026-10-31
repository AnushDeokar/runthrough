"""runthrough: evaluate deployed chat agents with simulated users, on Inspect AI."""

from importlib.metadata import PackageNotFoundError, version

from runthrough.adapters import AgentTurn, Target, ToolCallRecord, TurnContext

try:
    __version__ = version("runthrough")
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = "0.0.0"

__all__ = ["AgentTurn", "Target", "ToolCallRecord", "TurnContext", "__version__"]
