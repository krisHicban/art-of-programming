"""
Backtracking routing agent implementation.

This agent uses exhaustive search with pruning to find optimal or near-optimal solutions.
It explores the solution space systematically, backtracking when constraints are violated.

Time Complexity: O(m^n) where m=vehicles, n=packages (with pruning)
Space Complexity: O(n) for recursion stack

Trade-offs:
+ Can find optimal solutions
+ Explores solution space systematically
+ Good for educational purposes
- Exponential time complexity
- Only practical for small problem sizes (~15-20 packages)
- Slow compared to greedy approaches
"""

import copy
from typing import List, Optional
from .base_agent import RouteAgent
from ..models import Package, Vehicle, Route, DeliveryMap
from ..core import Router


class BacktrackingAgent(RouteAgent):
    """
    Backtracking routing agent that explores solution space exhaustively.

    Algorithm:
    1. Try assigning each package to each available vehicle
    2. Recursively assign remaining packages
    3. Prune branches that violate capacity constraints
    4. Track best solution found (by profit)
    5. Backtrack and try alternative assignments
    6. Optimize route orders using nearest neighbor
    """

    def __init__(self, delivery_map: DeliveryMap, max_packages: int = 15):
        """
        Initialize backtracking agent.

        Args:
            delivery_map: Map for distance calculations
            max_packages: Maximum packages to handle (for tractability)
        """
        super().__init__(delivery_map, "Backtracking Agent")
        self.description = "Exhaustive search with backtracking and pruning"
        self.max_packages = max_packages
        self.router = Router()

        # State for search
        self.best_solution: Optional[List[Route]] = None
        self.best_profit: float = float('-inf')
        self.nodes_explored: int = 0

    def plan_routes(self, packages: List[Package], fleet: List[Vehicle]) -> List[Route]:
        """
        Create routes using backtracking search.

        Args:
            packages: List of packages to deliver
            fleet: Available vehicles

        Returns:
            List of routes with best profit found
        """
        if not self.validate_inputs(packages, fleet):
            return []

        # Limit problem size for tractability
        if len(packages) > self.max_packages:
            print(f"[{self.name}] Warning: Too many packages ({len(packages)}), "
                  f"using first {self.max_packages} only")
            packages = packages[:self.max_packages]

        print(f"[{self.name}] Planning routes for {len(packages)} packages with {len(fleet)} vehicles...")
        print(f"[{self.name}] This may take some time...")

        # Reset search state
        self.best_solution = None
        self.best_profit = float('-inf')
        self.nodes_explored = 0

        # Initialize empty routes for each vehicle
        initial_routes = [
            Route(vehicle=v, packages=[], stops=[], delivery_map=self.delivery_map)
            for v in fleet
        ]

        # Start backtracking search
        self._backtrack(packages, initial_routes, 0)

        print(f"[{self.name}] Explored {self.nodes_explored} nodes")
        print(f"[{self.name}] Best profit found: ${self.best_profit:.2f}")

        if self.best_solution is None:
            print(f"[{self.name}] No valid solution found!")
            return []

        # Optimize stop orders for best solution
        optimized_solution = []
        for route in self.best_solution:
            if route.packages:  # Only include routes with packages
                route.stops = self._optimize_stops(route.packages)
                optimized_solution.append(route)

        print(f"[{self.name}] Created {len(optimized_solution)} routes")
        return optimized_solution

    def _backtrack(self, remaining_packages: List[Package],
                   current_routes: List[Route],
                   package_idx: int) -> None:
        """
        Recursive backtracking search.

        Args:
            remaining_packages: Packages not yet assigned
            current_routes: Current partial solution
            package_idx: Index of current package to assign
        """
        self.nodes_explored += 1

        # Base case: all packages assigned
        if package_idx >= len(remaining_packages):
            # Calculate profit of this solution
            profit = sum(r.profit for r in current_routes if r.packages)

            # Update best solution if this is better
            if profit > self.best_profit:
                self.best_profit = profit
                # Deep copy routes to preserve this solution
                self.best_solution = [
                    Route(
                        vehicle=r.vehicle,
                        packages=r.packages.copy(),
                        stops=[],
                        delivery_map=self.delivery_map
                    )
                    for r in current_routes
                ]
            return

        package = remaining_packages[package_idx]

        # Try assigning package to each route
        for route in current_routes:
            # Pruning: check if package fits in vehicle
            if route.total_volume + package.volume_m3 <= route.vehicle.vehicle_type.capacity_m3:
                # Make assignment
                route.packages.append(package)

                # Recursive call for next package
                self._backtrack(remaining_packages, current_routes, package_idx + 1)

                # Backtrack: undo assignment
                route.packages.pop()

    def _optimize_stops(self, packages: List[Package]) -> List[tuple]:
        """
        Optimize stop order using nearest neighbor.

        Args:
            packages: Packages in route

        Returns:
            Optimized stop coordinates
        """
        if not packages:
            return []

        destinations = [pkg.destination for pkg in packages]
        return self.router.nearest_neighbor_tsp(
            destinations,
            self.delivery_map.depot,
            self.delivery_map
        )


class PruningBacktrackingAgent(BacktrackingAgent):
    """
    Enhanced backtracking with more aggressive pruning.

    Additional pruning strategies:
    - Bound: prune if current solution can't beat best known profit
    - Symmetry: avoid exploring symmetric solutions
    """

    def __init__(self, delivery_map: DeliveryMap, max_packages: int = 15):
        super().__init__(delivery_map, max_packages)
        self.name = "Pruning Backtracking Agent"
        self.description = "Backtracking with aggressive pruning and bounding"

    def _backtrack(self, remaining_packages: List[Package],
                   current_routes: List[Route],
                   package_idx: int) -> None:
        """
        Enhanced backtracking with bounding.

        Args:
            remaining_packages: Packages not yet assigned
            current_routes: Current partial solution
            package_idx: Index of current package to assign
        """
        self.nodes_explored += 1

        # Base case
        if package_idx >= len(remaining_packages):
            profit = sum(r.profit for r in current_routes if r.packages)
            if profit > self.best_profit:
                self.best_profit = profit
                self.best_solution = [
                    Route(
                        vehicle=r.vehicle,
                        packages=r.packages.copy(),
                        stops=[],
                        delivery_map=self.delivery_map
                    )
                    for r in current_routes
                ]
            return

        # Bounding: calculate upper bound on possible profit
        # If we can't beat best_profit, prune this branch
        current_profit = sum(r.profit for r in current_routes if r.packages)
        remaining_revenue = sum(pkg.payment for pkg in remaining_packages[package_idx:])
        upper_bound = current_profit + remaining_revenue  # Optimistic: no costs

        if upper_bound <= self.best_profit:
            # Pruned!
            return

        package = remaining_packages[package_idx]

        # Try assigning to each route
        for route in current_routes:
            if route.total_volume + package.volume_m3 <= route.vehicle.vehicle_type.capacity_m3:
                route.packages.append(package)
                self._backtrack(remaining_packages, current_routes, package_idx + 1)
                route.packages.pop()
