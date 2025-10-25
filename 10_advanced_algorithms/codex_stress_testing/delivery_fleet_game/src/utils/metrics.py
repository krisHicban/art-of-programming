"""Helpers for computing performance metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from models.route import Route


@dataclass
class RouteMetrics:
    """Aggregated metrics across one or more routes."""

    total_distance: float
    total_revenue: float
    total_cost: float
    total_profit: float
    vehicle_count: int
    package_count: int


def aggregate_routes(routes: Iterable[Route]) -> RouteMetrics:
    total_distance = 0.0
    total_revenue = 0.0
    total_cost = 0.0
    package_count = 0
    vehicle_count = 0

    for route in routes:
        vehicle_count += 1
        package_count += len(route.package_ids)
        total_distance += route.total_distance_km
        total_revenue += route.total_revenue
        total_cost += route.total_cost

    return RouteMetrics(
        total_distance=total_distance,
        total_revenue=total_revenue,
        total_cost=total_cost,
        total_profit=total_revenue - total_cost,
        vehicle_count=vehicle_count,
        package_count=package_count,
    )
