"""Validation tests for planned routes."""

import pytest

from core.route_builder import build_route  # type: ignore
from core.validator import ValidationError, validate_routes  # type: ignore
from models.package import Package  # type: ignore
from models.vehicle import Vehicle  # type: ignore


def make_vehicle(capacity=10, range_km=None) -> Vehicle:
    return Vehicle(
        id="veh_1",
        type="small_van",
        capacity_m3=capacity,
        cost_per_km=1.0,
        purchase_price=10000,
        max_range_km=range_km,
    )


def make_package(volume=2.0, destination=(5, 0), package_id="pkg_1") -> Package:
    return Package(
        id=package_id,
        volume_m3=volume,
        payment_received=40,
        destination=destination,
    )


def test_validate_routes_capacity_violation() -> None:
    vehicle = make_vehicle(capacity=1.0)
    package = make_package(volume=3.0)
    route = build_route(vehicle, [package], depot=(0, 0))
    with pytest.raises(ValidationError):
        validate_routes([route], {vehicle.id: vehicle}, depot=(0, 0))


def test_validate_routes_success() -> None:
    vehicle = make_vehicle(capacity=10.0, range_km=50.0)
    packages = [
        make_package(destination=(3, 4), package_id="pkg_a"),
        make_package(volume=1.0, destination=(0, 4), package_id="pkg_b"),
    ]
    route = build_route(vehicle, packages, depot=(0, 0))
    validate_routes([route], {vehicle.id: vehicle}, depot=(0, 0))
