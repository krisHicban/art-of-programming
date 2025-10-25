"""Tests for routing utilities."""

from core.route_builder import build_route  # type: ignore
from core.router import euclidean_distance, route_distance  # type: ignore
from models.package import Package  # type: ignore
from models.vehicle import Vehicle  # type: ignore


def test_euclidean_distance_zero() -> None:
    assert euclidean_distance((0, 0), (0, 0)) == 0


def test_route_distance_sum() -> None:
    path = [(0, 0), (3, 4), (3, 0)]
    assert route_distance(path) == 5 + 4


def test_build_route_includes_depot() -> None:
    vehicle = Vehicle(
        id="veh_01",
        type="small_van",
        capacity_m3=10,
        cost_per_km=1.0,
        purchase_price=10000,
    )
    packages = [
        Package(id="pkg_a", volume_m3=2, payment_received=50, destination=(3, 4)),
        Package(id="pkg_b", volume_m3=1, payment_received=30, destination=(6, 8)),
    ]
    route = build_route(vehicle, packages, depot=(0, 0))
    assert route.stops[0] == (0, 0)
    assert route.stops[-1] == (0, 0)
    assert route.total_volume == 3
    assert route.total_revenue == 80
