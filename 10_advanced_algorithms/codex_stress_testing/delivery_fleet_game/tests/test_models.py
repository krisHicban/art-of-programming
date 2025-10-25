"""Smoke tests for model scaffolding."""

from models.package import Package  # type: ignore
from models.vehicle import Vehicle  # type: ignore


def test_vehicle_fields_round_trip() -> None:
    vehicle = Vehicle(
        id="veh_001",
        type="small_van",
        capacity_m3=10,
        cost_per_km=0.5,
        purchase_price=15000,
        max_range_km=200,
    )
    assert vehicle.current_location == (0.0, 0.0)
    vehicle.current_location = (5, 5)
    vehicle.reset_location()
    assert vehicle.current_location == (0.0, 0.0)


def test_package_defaults_origin_depot() -> None:
    pkg = Package(
        id="pkg_001",
        volume_m3=3.2,
        payment_received=80,
        destination=(10, -5),
    )
    assert pkg.origin == (0.0, 0.0)
