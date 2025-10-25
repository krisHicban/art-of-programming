"""Tests for agent scaffolding."""

from agents.base_agent import AgentContext  # type: ignore
from agents.greedy_agent import GreedyAgent  # type: ignore
from models.map import MapConfig  # type: ignore
from models.package import Package  # type: ignore
from models.vehicle import Vehicle  # type: ignore


def build_context(packages, vehicles) -> AgentContext:
    return AgentContext(
        day=1,
        balance=100_000.0,
        map_config=MapConfig(width=100, height=100, depot=(0.0, 0.0)),
        vehicles=vehicles,
        pending_packages=packages,
    )


def test_greedy_agent_assigns_within_capacity() -> None:
    agent = GreedyAgent()
    vehicles = [
        Vehicle(id="veh_a", type="small_van", capacity_m3=10, cost_per_km=0.5, purchase_price=10000),
        Vehicle(id="veh_b", type="medium_truck", capacity_m3=25, cost_per_km=0.8, purchase_price=20000),
    ]
    packages = [
        Package(id="pkg_1", volume_m3=5, payment_received=50, destination=(5, 5)),
        Package(id="pkg_2", volume_m3=3, payment_received=40, destination=(2, 3)),
        Package(id="pkg_3", volume_m3=2, payment_received=30, destination=(-4, 1)),
    ]
    context = build_context(packages, vehicles)
    plan = agent.plan(context)
    assert plan.success
    assigned_ids = {pkg.id for pkgs in plan.assignments.values() for pkg in pkgs}
    assert assigned_ids == {"pkg_1", "pkg_2", "pkg_3"}
    assert not plan.unassigned


def test_greedy_agent_reports_unassigned_when_over_capacity() -> None:
    agent = GreedyAgent()
    vehicles = [
        Vehicle(id="veh_a", type="small_van", capacity_m3=4, cost_per_km=0.5, purchase_price=10000),
    ]
    packages = [
        Package(id="pkg_1", volume_m3=3, payment_received=50, destination=(5, 5)),
        Package(id="pkg_2", volume_m3=3, payment_received=40, destination=(2, 3)),
    ]
    context = build_context(packages, vehicles)
    plan = agent.plan(context)
    assert not plan.success
    assert len(plan.unassigned) == 1


def test_greedy_agent_handles_many_packages() -> None:
    agent = GreedyAgent()
    vehicles = [
        Vehicle(id="veh_a", type="small_van", capacity_m3=10, cost_per_km=0.5, purchase_price=10000),
        Vehicle(id="veh_b", type="medium_truck", capacity_m3=25, cost_per_km=0.8, purchase_price=20000),
        Vehicle(id="veh_c", type="large_truck", capacity_m3=50, cost_per_km=1.2, purchase_price=40000),
    ]
    packages = [
        Package(
            id=f"pkg_{i}",
            volume_m3=volume,
            payment_received=50 + i,
            destination=(i * 2.0, i * -1.5),
        )
        for i, volume in enumerate([12, 8, 5, 3, 7, 9, 4, 6, 10, 2], start=1)
    ]
    context = build_context(packages, vehicles)
    plan = agent.plan(context)
    assert plan.success
    assert sum(len(pkgs) for pkgs in plan.assignments.values()) == len(packages)
