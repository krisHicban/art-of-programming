"""Smoke tests for JSON data loading."""

from pathlib import Path

from models.game_state import AgentRun  # type: ignore
from models.package import Package  # type: ignore
from utils import data_loader  # type: ignore


def test_load_vehicle_catalog_defaults() -> None:
    catalog = data_loader.load_vehicle_catalog()
    assert "small_van" in catalog
    assert catalog["small_van"]["capacity_m3"] == 10


def test_load_packages_for_day() -> None:
    # Use packaged data file to ensure loader works end-to-end.
    packages = data_loader.load_packages_for_day(1)
    assert len(packages) >= 1
    destinations = {pkg.destination for pkg in packages}
    assert (15.0, 25.0) in destinations


def test_save_and_load_game_state_preserves_packages(tmp_path: Path) -> None:
    catalog = data_loader.load_vehicle_catalog()
    state = data_loader.load_game_state(catalog)
    pending_pkg = Package(
        id="pkg_pending",
        volume_m3=1.0,
        payment_received=25.0,
        destination=(1.0, 2.0),
    )
    in_transit_pkg = Package(
        id="pkg_transit",
        volume_m3=3.5,
        payment_received=60.0,
        destination=(-3.0, 4.0),
    )
    delivered_pkg = Package(
        id="pkg_delivered",
        volume_m3=2.5,
        payment_received=45.0,
        destination=(0.0, -5.0),
    )
    state.packages_pending.append(pending_pkg)
    state.packages_in_transit.append(in_transit_pkg)
    state.packages_delivered.append(delivered_pkg)
    state.record_agent_run(
        AgentRun(
            day=state.current_day,
            agent_name="Test Agent",
            success=True,
            packages_assigned=1,
            packages_unassigned=0,
            total_distance=10.0,
            total_revenue=100.0,
            total_cost=30.0,
            total_profit=70.0,
        )
    )
    state.log_event(
        phase="planning",
        event_type="test_event",
        description="Testing event logging.",
        payload={"key": "value"},
    )

    save_path = tmp_path / "savegame.json"
    data_loader.save_game_state(state, save_path)

    restored = data_loader.load_game_state(catalog, path=save_path)
    restored_pending_ids = {pkg.id for pkg in restored.packages_pending}
    restored_transit_ids = {pkg.id for pkg in restored.packages_in_transit}
    restored_delivered_ids = {pkg.id for pkg in restored.packages_delivered}

    assert "pkg_pending" in restored_pending_ids
    assert "pkg_transit" in restored_transit_ids
    assert "pkg_delivered" in restored_delivered_ids
    assert restored.agent_history
    run = restored.agent_history[-1]
    assert run.agent_name == "Test Agent"
    assert run.total_profit == 70.0
    assert restored.events
    assert restored.events[-1].event_type == "test_event"
