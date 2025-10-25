"""Vehicle domain model."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Vehicle:
    """Represents a delivery vehicle and its operational attributes."""

    id: str
    type: str
    capacity_m3: float
    cost_per_km: float
    purchase_price: float
    max_range_km: Optional[float] = None
    current_location: Point = field(default_factory=lambda: (0.0, 0.0))

    def reset_location(self) -> None:
        """Return vehicle to depot after completing a route."""
        self.current_location = (0.0, 0.0)
