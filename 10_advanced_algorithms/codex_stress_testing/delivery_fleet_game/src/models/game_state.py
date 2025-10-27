"""Game state container tracking simulation progress."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .vehicle import Vehicle
from .package import Package


@dataclass
class DailySummary:
    """Aggregated metrics for a single day."""

    day: int
    packages_delivered: int
    revenue: float
    costs: float
    profit: float


@dataclass
class AgentRun:
    """Metrics captured when an agent produces a plan."""

    day: int
    agent_name: str
    success: bool
    packages_assigned: int
    packages_unassigned: int
    total_distance: float
    total_revenue: float
    total_cost: float
    total_profit: float
    notes: Optional[str] = None


@dataclass
class GameEvent:
    """A discrete event captured during the simulation for visualization or auditing."""

    timestamp: str
    day: int
    phase: str
    event_type: str
    description: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    """Minimal placeholder for evolving game state."""

    current_day: int = 1
    balance: float = 100_000.0
    fleet: List[Vehicle] = field(default_factory=list)
    packages_pending: List[Package] = field(default_factory=list)
    packages_in_transit: List[Package] = field(default_factory=list)
    packages_delivered: List[Package] = field(default_factory=list)
    daily_history: List[DailySummary] = field(default_factory=list)
    agent_history: List[AgentRun] = field(default_factory=list)
    events: List[GameEvent] = field(default_factory=list)

    def advance_day(self) -> None:
        self.current_day += 1

    def record_summary(self, summary: DailySummary) -> None:
        self.daily_history.append(summary)
        self.balance += summary.profit

    def record_agent_run(self, run: AgentRun) -> None:
        self.agent_history.append(run)

    def log_event(
        self,
        phase: str,
        event_type: str,
        description: str,
        payload: Optional[Dict[str, Any]] = None,
        day_override: Optional[int] = None,
    ) -> None:
        from datetime import datetime, timezone

        timestamp = datetime.now(tz=timezone.utc).isoformat()
        self.events.append(
            GameEvent(
                timestamp=timestamp,
                day=day_override if day_override is not None else self.current_day,
                phase=phase,
                event_type=event_type,
                description=description,
                payload=payload or {},
            )
        )
