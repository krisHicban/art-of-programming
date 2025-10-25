"""Ledger calculations for daily summaries."""

from core.ledger import EntryType, Ledger, LedgerEntry  # type: ignore
from models.route import Route  # type: ignore


def test_ledger_profit_calculation() -> None:
    ledger = Ledger()
    ledger.record(LedgerEntry(EntryType.REVENUE, "Delivery income", 200))
    ledger.record(LedgerEntry(EntryType.EXPENSE, "Fuel costs", 50))
    assert ledger.revenue() == 200
    assert ledger.expenses() == 50
    assert ledger.profit() == 150


def test_ledger_record_route() -> None:
    ledger = Ledger()
    route = Route(
        vehicle_id="veh_1",
        package_ids=["pkg_1"],
        stops=[(0, 0), (1, 1), (0, 0)],
        total_distance_km=3.0,
        total_cost=15.0,
        total_revenue=60.0,
        total_volume=2.0,
    )
    ledger.record_route(route)
    assert ledger.revenue() == 60.0
    assert ledger.expenses() == 15.0
    assert ledger.profit() == 45.0
