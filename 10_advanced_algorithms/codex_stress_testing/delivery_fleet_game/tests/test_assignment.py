"""Assignment manager behavior."""

from core.assignment import AssignmentManager  # type: ignore
from models.package import Package  # type: ignore
from models.vehicle import Vehicle  # type: ignore


def make_vehicle(vehicle_id: str) -> Vehicle:
    return Vehicle(
        id=vehicle_id,
        type="small_van",
        capacity_m3=10,
        cost_per_km=0.5,
        purchase_price=15000,
    )


def make_package(package_id: str) -> Package:
    return Package(
        id=package_id,
        volume_m3=2.0,
        payment_received=30.0,
        destination=(5.0, 5.0),
    )


def test_assignment_add_and_remove() -> None:
    vehicles = [make_vehicle("veh_1")]
    manager = AssignmentManager(vehicles)
    package = make_package("pkg_1")

    manager.add_package("veh_1", package)
    assignment = manager.find_assignment_for_package("pkg_1")
    assert assignment is not None
    assert assignment.total_volume == 2.0
    assert manager.has_assignments()

    removed = manager.remove_package("veh_1", "pkg_1")
    assert removed is package
    assert manager.find_assignment_for_package("pkg_1") is None
    assert not manager.has_assignments()
