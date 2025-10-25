"""Route model linking vehicles to delivery stops."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Route:
    """Placeholder for route aggregation."""

    vehicle_id: str
    package_ids: List[str] = field(default_factory=list)
    stops: List[Point] = field(default_factory=list)
    total_distance_km: float = 0.0
    total_cost: float = 0.0
    total_revenue: float = 0.0
    total_volume: float = 0.0
    is_valid: bool = False
    notes: Optional[str] = None
