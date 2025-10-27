"""Controllers translating snapshots into view-ready models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json


@dataclass
class SnapshotData:
    """Lightweight representation of exported snapshot fields."""

    raw: Dict[str, Any]

    @property
    def events(self) -> List[Dict[str, Any]]:
        return self.raw.get("events", [])

    @property
    def agent_history(self) -> List[Dict[str, Any]]:
        return self.raw.get("agent_history", [])


def load_snapshot(path: Path) -> SnapshotData:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return SnapshotData(raw=payload)
