"""Financial ledger utilities for tracking income and expenses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

from models.route import Route


class EntryType(Enum):
    """Types of ledger entries."""

    REVENUE = auto()
    EXPENSE = auto()


@dataclass
class LedgerEntry:
    """Single financial transaction."""

    entry_type: EntryType
    label: str
    amount: float

    @property
    def signed_amount(self) -> float:
        """Return positive amount for revenue, negative for expense."""
        if self.entry_type == EntryType.REVENUE:
            return self.amount
        return -self.amount


class Ledger:
    """Transaction list used to calculate daily profit/loss."""

    def __init__(self) -> None:
        self.entries: List[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)

    def record_route(self, route: Route) -> None:
        """Log revenue and cost entries for a single route."""
        if route.total_revenue:
            self.record(LedgerEntry(EntryType.REVENUE, f"Route revenue ({route.vehicle_id})", route.total_revenue))
        if route.total_cost:
            self.record(LedgerEntry(EntryType.EXPENSE, f"Route cost ({route.vehicle_id})", route.total_cost))

    def revenue(self) -> float:
        return sum(entry.amount for entry in self.entries if entry.entry_type == EntryType.REVENUE)

    def expenses(self) -> float:
        return sum(entry.amount for entry in self.entries if entry.entry_type == EntryType.EXPENSE)

    def profit(self) -> float:
        return sum(entry.signed_amount for entry in self.entries)
