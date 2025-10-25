"""Package domain model."""

from dataclasses import dataclass, field
from typing import Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Package:
    """Represents a package awaiting delivery."""

    id: str
    volume_m3: float
    payment_received: float
    destination: Point
    origin: Point = field(default_factory=lambda: (0.0, 0.0))
    priority: Optional[int] = None
    weight_kg: Optional[float] = None
    received_date: Optional[int] = None
    delivery_deadline: Optional[int] = None
