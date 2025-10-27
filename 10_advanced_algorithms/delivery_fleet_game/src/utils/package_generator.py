"""
Dynamic package generator for delivery simulation.

Generates packages based on marketing level and target volume.
"""

import random
from typing import List, Tuple
from ..models.package import Package
from ..models.map import DeliveryMap


class PackageGenerator:
    """
    Generates packages dynamically based on target volume.

    Creates realistic package distributions with varying sizes,
    destinations, and payments.
    """

    def __init__(self, delivery_map: DeliveryMap, seed: int = None):
        """
        Initialize package generator.

        Args:
            delivery_map: Map for calculating distances and destinations
            seed: Random seed for reproducibility (optional)
        """
        self.delivery_map = delivery_map
        if seed is not None:
            random.seed(seed)

    def generate_packages(self, target_volume: float, day: int) -> List[Package]:
        """
        Generate packages with target total volume.

        Args:
            target_volume: Target total volume in m³
            day: Current day number (for package IDs)

        Returns:
            List of generated packages
        """
        packages = []
        current_volume = 0.0
        package_count = 0

        # Calculate number of packages (average 2.5m³ per package)
        avg_package_size = 2.5
        num_packages = max(8, int(target_volume / avg_package_size))

        # Package size distribution: 40% small, 40% medium, 20% large
        size_distribution = [
            ('small', 1.0, 2.5, 0.4),   # 1.0-2.5m³, 40%
            ('medium', 2.5, 4.0, 0.4),  # 2.5-4.0m³, 40%
            ('large', 4.0, 6.0, 0.2),   # 4.0-6.0m³, 20%
        ]

        while current_volume < target_volume and package_count < num_packages * 1.5:
            # Select size category
            remaining_volume = target_volume - current_volume

            # Choose size based on remaining volume and distribution
            size_type = self._choose_size_category(size_distribution, remaining_volume)
            volume = self._generate_volume(size_type, size_distribution, remaining_volume)

            # Generate destination
            destination = self._generate_destination()

            # Calculate payment based on distance and volume
            distance = self.delivery_map.distance(self.delivery_map.depot, destination)
            payment = self._calculate_payment(volume, distance)

            # Determine priority (15% are high priority)
            priority = 3 if random.random() < 0.15 else 1
            if priority == 3:
                payment *= 1.5  # High priority pays more

            # Create package
            package_id = f"pkg_d{day}_{package_count + 1:03d}"
            description = self._generate_description(volume, distance)

            package = Package(
                id=package_id,
                destination=destination,
                volume_m3=round(volume, 1),
                payment=round(payment, 2),
                priority=priority,
                description=description,
                received_day=day
            )

            packages.append(package)
            current_volume += volume
            package_count += 1

            # Stop if we're close enough to target
            if current_volume >= target_volume * 0.95:
                break

        print(f"Generated {len(packages)} packages, {current_volume:.1f}m³ (target: {target_volume:.1f}m³)")
        return packages

    def _choose_size_category(self, distribution: List[Tuple], remaining_volume: float) -> str:
        """Choose package size category based on distribution and remaining volume."""
        # If remaining volume is small, prefer smaller packages
        if remaining_volume < 3.0:
            return 'small'
        elif remaining_volume < 5.0:
            choices = ['small', 'medium']
            return random.choice(choices)

        # Normal distribution-based selection
        rand = random.random()
        cumulative = 0.0
        for size_type, min_vol, max_vol, probability in distribution:
            cumulative += probability
            if rand <= cumulative:
                return size_type

        return 'medium'  # Default

    def _generate_volume(self, size_type: str, distribution: List[Tuple], remaining_volume: float) -> float:
        """Generate volume for a package of given size type."""
        # Find size range
        for st, min_vol, max_vol, _ in distribution:
            if st == size_type:
                # Generate random volume in range, respecting remaining volume
                max_vol = min(max_vol, remaining_volume + 0.5)
                volume = random.uniform(min_vol, max_vol)
                return volume

        return 2.0  # Default

    def _generate_destination(self) -> Tuple[float, float]:
        """Generate random destination on the map."""
        # Cluster destinations in sectors for more realistic routing
        # 70% in main area, 30% in outer areas

        if random.random() < 0.7:
            # Main delivery area (closer to depot)
            x = random.uniform(10, 60)
            y = random.uniform(10, 60)
        else:
            # Outer delivery area
            x = random.uniform(5, 95)
            y = random.uniform(5, 95)

        return (round(x, 1), round(y, 1))

    def _calculate_payment(self, volume: float, distance: float) -> float:
        """
        Calculate payment for package based on volume and distance.

        Payment formula: base_rate * volume + distance_rate * distance
        """
        base_rate = 15.0  # $/m³
        distance_rate = 0.3  # $/km

        payment = base_rate * volume + distance_rate * distance

        # Add randomness (±10%)
        payment *= random.uniform(0.9, 1.1)

        return payment

    def _generate_description(self, volume: float, distance: float) -> str:
        """Generate descriptive text for package."""
        items = [
            "Electronics", "Furniture", "Documents", "Clothing",
            "Hardware", "Books", "Appliances", "Toys",
            "Medical Supplies", "Office Equipment", "Food Items",
            "Auto Parts", "Sports Equipment", "Art Supplies"
        ]

        item = random.choice(items)

        if volume < 2.0:
            size = "Small"
        elif volume < 4.0:
            size = "Medium"
        else:
            size = "Large"

        return f"{size} {item}"
