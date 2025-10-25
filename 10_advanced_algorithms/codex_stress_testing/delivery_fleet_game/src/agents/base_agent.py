"""Base classes and shared structures for routing agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List

from models.map import MapConfig
from models.package import Package
from models.vehicle import Vehicle


@dataclass
class AgentContext:
    """Snapshot of the planning state passed into agents."""

    day: int
    balance: float
    map_config: MapConfig
    vehicles: List[Vehicle]
    pending_packages: List[Package]


@dataclass
class AgentPlan:
    """Result returned by an agent after route planning."""

    assignments: Dict[str, List[Package]] = field(default_factory=dict)
    unassigned: List[Package] = field(default_factory=list)
    notes: str = ""
    success: bool = True


class BaseAgent(ABC):
    """Abstract base class for all route-planning agents."""

    name: str = "Base Agent"

    @abstractmethod
    def plan(self, context: AgentContext) -> AgentPlan:
        """Produce an assignment plan for the provided context."""

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.name
