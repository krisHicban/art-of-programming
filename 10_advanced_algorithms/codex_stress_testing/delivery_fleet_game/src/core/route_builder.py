"""Utilities for constructing route objects from package assignments."""

from __future__ import annotations

from typing import Callable, Sequence

from core.router import euclidean_distance, route_distance
from models.package import Package
from models.route import Route
from models.vehicle import Vehicle

Point = tuple[float, float]
DistanceMetric = Callable[[Point, Point], float]


def build_route(
    vehicle: Vehicle,
    packages: Sequence[Package],
    depot: Point,
    metric: DistanceMetric = euclidean_distance,
) -> Route:
    """Create a Route with computed totals for distance, volume, and cost."""
    if not packages:
        raise ValueError("Cannot build route without packages")

    waypoints = [depot, *[pkg.destination for pkg in packages], depot]
    total_distance = route_distance(waypoints, metric)
    total_volume = sum(pkg.volume_m3 for pkg in packages)
    total_revenue = sum(pkg.payment_received for pkg in packages)
    total_cost = total_distance * vehicle.cost_per_km

    return Route(
        vehicle_id=vehicle.id,
        package_ids=[pkg.id for pkg in packages],
        stops=waypoints,
        total_distance_km=total_distance,
        total_cost=total_cost,
        total_revenue=total_revenue,
        total_volume=total_volume,
    )
