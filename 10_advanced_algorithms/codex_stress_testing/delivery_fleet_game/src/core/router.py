"""Routing helpers and distance calculations."""

from typing import Iterable, Tuple, Optional


Point = Tuple[float, float]


def euclidean_distance(a: Point, b: Point) -> float:
    """Compute Euclidean distance between two map points."""
    ax, ay = a
    bx, by = b
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def manhattan_distance(a: Point, b: Point) -> float:
    """Compute Manhattan distance between two map points."""
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by)


def route_distance(points: Iterable[Point], metric=euclidean_distance) -> float:
    """Calculate total distance across consecutive points using the provided metric."""
    total = 0.0
    previous: Optional[Point] = None
    for point in points:
        if previous is not None:
            total += metric(previous, point)
        previous = point
    return total
