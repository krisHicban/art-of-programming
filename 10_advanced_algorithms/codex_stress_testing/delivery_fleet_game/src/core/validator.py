"""Constraint validation for routes and assignments."""

from __future__ import annotations

from typing import Iterable, Mapping, Set

from core.router import euclidean_distance
from models.route import Route
from models.vehicle import Vehicle

Point = tuple[float, float]


class ValidationError(Exception):
    """Raised when a route or assignment violates constraints."""


def validate_routes(
    routes: Iterable[Route],
    vehicles: Mapping[str, Vehicle],
    depot: Point,
    metric=euclidean_distance,
) -> None:
    """Validate a collection of routes against capacity, depot, and uniqueness constraints."""
    seen_packages: Set[str] = set()
    for route in routes:
        vehicle = vehicles.get(route.vehicle_id)
        if vehicle is None:
            raise ValidationError(f"Route references unknown vehicle '{route.vehicle_id}'")
        _ensure_capacity(route, vehicle)
        _ensure_depot_loop(route, depot)
        _ensure_within_range(route, vehicle)

        expected_distance = _recompute_distance(route, metric)
        if abs(expected_distance - route.total_distance_km) > 1e-6:
            raise ValidationError(
                f"Route distance mismatch for vehicle {vehicle.id}: "
                f"{route.total_distance_km:.2f} recorded vs {expected_distance:.2f} computed"
            )

        for package_id in route.package_ids:
            if package_id in seen_packages:
                raise ValidationError(f"Package '{package_id}' assigned multiple times.")
            seen_packages.add(package_id)


def _ensure_capacity(route: Route, vehicle: Vehicle) -> None:
    if route.total_volume > vehicle.capacity_m3 + 1e-6:
        raise ValidationError(
            f"Route for vehicle {vehicle.id} exceeds capacity. "
            f"{route.total_volume:.2f} m3 > {vehicle.capacity_m3:.2f} m3"
        )


def _ensure_within_range(route: Route, vehicle: Vehicle) -> None:
    if vehicle.max_range_km is None:
        return
    if route.total_distance_km > vehicle.max_range_km + 1e-6:
        raise ValidationError(
            f"Route for vehicle {vehicle.id} exceeds max range. "
            f"{route.total_distance_km:.2f} km > {vehicle.max_range_km:.2f} km"
        )


def _ensure_depot_loop(route: Route, depot: Point) -> None:
    if not route.stops:
        raise ValidationError("Route contains no stops.")
    if route.stops[0] != depot:
        raise ValidationError("Route must start at the depot.")
    if route.stops[-1] != depot:
        raise ValidationError("Route must end at the depot.")


def _recompute_distance(route: Route, metric) -> float:
    total = 0.0
    previous = None
    for point in route.stops:
        if previous is not None:
            total += metric(previous, point)
        previous = point
    return total
