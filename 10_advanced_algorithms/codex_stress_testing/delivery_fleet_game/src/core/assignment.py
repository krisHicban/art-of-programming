"""Manual assignment workflow for Phase 1 planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.route_builder import build_route
from core.router import euclidean_distance

from models.package import Package
from models.route import Route
from models.vehicle import Vehicle


@dataclass
class Assignment:
    """Represents packages assigned to a particular vehicle prior to validation."""

    vehicle: Vehicle
    packages: List[Package] = field(default_factory=list)

    @property
    def total_volume(self) -> float:
        return sum(pkg.volume_m3 for pkg in self.packages)

    @property
    def total_revenue(self) -> float:
        return sum(pkg.payment_received for pkg in self.packages)


class AssignmentManager:
    """Coordinates manual assignment steps for the CLI flow."""

    def __init__(self, vehicles: List[Vehicle]) -> None:
        self._assignments: Dict[str, Assignment] = {
            vehicle.id: Assignment(vehicle=vehicle) for vehicle in vehicles
        }

    def add_package(self, vehicle_id: str, package: Package) -> None:
        assignment = self._assignments.get(vehicle_id)
        if assignment is None:
            raise KeyError(f"Unknown vehicle id '{vehicle_id}'")
        assignment.packages.append(package)

    def remove_package(self, vehicle_id: str, package_id: str) -> Package | None:
        assignment = self._assignments.get(vehicle_id)
        if assignment is None:
            raise KeyError(f"Unknown vehicle id '{vehicle_id}'")
        for index, package in enumerate(assignment.packages):
            if package.id == package_id:
                return assignment.packages.pop(index)
        return None

    def find_assignment_for_package(self, package_id: str) -> Assignment | None:
        for assignment in self._assignments.values():
            if any(pkg.id == package_id for pkg in assignment.packages):
                return assignment
        return None

    def build_routes(self, depot: tuple[float, float], metric=euclidean_distance) -> List[Route]:
        """Convert assignments into Route objects for validation and execution."""
        routes: List[Route] = []
        for assignment in self._assignments.values():
            if not assignment.packages:
                continue
            route = build_route(
                vehicle=assignment.vehicle,
                packages=assignment.packages,
                depot=depot,
                metric=metric,
            )
            routes.append(route)
        return routes

    def clear(self) -> None:
        for assignment in self._assignments.values():
            assignment.packages.clear()

    def summary(self) -> List[Assignment]:
        return list(self._assignments.values())

    def has_assignments(self) -> bool:
        return any(assignment.packages for assignment in self._assignments.values())
