"""Map configuration model."""

from dataclasses import dataclass, field
from typing import List, Tuple

Point = Tuple[float, float]


@dataclass
class Location:
    """Named coordinate on the map."""

    id: str
    name: str
    x: float
    y: float

    @property
    def as_point(self) -> Point:
        return (self.x, self.y)


@dataclass
class MapConfig:
    """Holds map dimensions and named locations."""

    width: float
    height: float
    depot: Point
    locations: List[Location] = field(default_factory=list)

    def within_bounds(self, point: Point) -> bool:
        """Determine whether a point lies within the map bounds."""
        x, y = point
        return -self.width <= x <= self.width and -self.height <= y <= self.height

    def get_location(self, location_id: str) -> Location:
        for location in self.locations:
            if location.id == location_id:
                return location
        raise KeyError(f"Unknown location id: {location_id}")
