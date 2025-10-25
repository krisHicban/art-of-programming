"""Game state container tracking simulation progress."""

from dataclasses import dataclass, field
from typing import List

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
class GameState:
    """Minimal placeholder for evolving game state."""

    current_day: int = 1
    balance: float = 100_000.0
    fleet: List[Vehicle] = field(default_factory=list)
    packages_pending: List[Package] = field(default_factory=list)
    packages_in_transit: List[Package] = field(default_factory=list)
    packages_delivered: List[Package] = field(default_factory=list)
    daily_history: List[DailySummary] = field(default_factory=list)

    def advance_day(self) -> None:
        self.current_day += 1

    def record_summary(self, summary: DailySummary) -> None:
        self.daily_history.append(summary)
        self.balance += summary.profit
