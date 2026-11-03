"""Your own pass/fail labels for judged conversations, stored as JSON Lines.

The file is append-only: changing a label appends a new line, and the latest line
for an item wins. That keeps a history and makes concurrent tools safe to use.
"""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

LabelValue = Literal["pass", "fail", "skip"]


class Label(BaseModel):
    key: str
    """``<sample uuid>:<scorer name>``; identifies one judged conversation."""
    scorer: str
    sample_id: str
    epoch: int
    label: LabelValue
    note: str = ""
    labeled_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class LabelStore:
    """Read and write labels in a ``.jsonl`` file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> dict[str, Label]:
        """Latest label per item key. A missing file means no labels yet."""
        labels: dict[str, Label] = {}
        if not self.path.exists():
            return labels
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                label = Label.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{self.path}:{number}: not a valid label: {exc}") from exc
            labels[label.key] = label
        return labels

    def add(self, label: Label) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(label.model_dump_json() + "\n")
