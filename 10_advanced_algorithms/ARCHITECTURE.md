# Delivery Fleet System - Technical Architecture

## Design Philosophy

This project demonstrates professional software engineering practices while teaching algorithmic concepts:

1. **Separation of Concerns:** Models, logic, UI, and algorithms are independent
2. **Strategy Pattern:** Agents are interchangeable strategies
3. **SOLID Principles:** Clean, maintainable, extensible code
4. **Type Safety:** Full type hints for better IDE support and error prevention
5. **Testability:** Pure functions and dependency injection for easy testing

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│                    (Entry Point / CLI)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │      GameEngine (core)         │
         │  - Day cycle management        │
         │  - State transitions           │
         │  - Agent orchestration         │
         └───────┬───────────────┬────────┘
                 │               │
        ┌────────▼─────┐    ┌───▼──────────┐
        │  GameState   │    │   Router     │
        │  (models)    │    │   (core)     │
        └──────┬───────┘    └──────────────┘
               │
    ┌──────────┼──────────┬──────────┬─────────┐
    ▼          ▼          ▼          ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌──────┐ ┌──────┐
│Vehicle │ │Package │ │ Route  │ │ Map  │ │Fleet │
└────────┘ └────────┘ └────────┘ └──────┘ └──────┘

               ┌─────────────────────┐
               │   Agent (Strategy)   │
               │   <<abstract>>       │
               └──────────┬───────────┘
                          │
        ┌─────────────────┼─────────────────┬──────────┐
        ▼                 ▼                 ▼          ▼
  ┌──────────┐    ┌──────────────┐   ┌──────────┐  ┌─────┐
  │  Greedy  │    │ Backtracking │   │    DP    │  │ ... │
  │  Agent   │    │    Agent     │   │  Agent   │  └─────┘
  └──────────┘    └──────────────┘   └──────────┘

               ┌─────────────────────┐
               │   UI Layer (Pygame)  │
               │  - MapRenderer       │
               │  - StatsPanel        │
               │  - ControlPanel      │
               └─────────────────────┘

               ┌─────────────────────┐
               │  Data Layer (JSON)   │
               │  - DataLoader        │
               │  - SaveManager       │
               └─────────────────────┘
```

---

## Core Components

### 1. Models Layer (`src/models/`)

#### `vehicle.py`
```python
@dataclass
class VehicleType:
    name: str
    capacity_m3: float
    cost_per_km: float
    purchase_price: float
    max_range_km: float

@dataclass
class Vehicle:
    id: str
    vehicle_type: VehicleType
    current_location: tuple[float, float] = (0, 0)  # Depot

    def can_carry(self, volume: float) -> bool:
        """Check if vehicle can carry given volume"""

    def calculate_trip_cost(self, distance_km: float) -> float:
        """Calculate operating cost for distance"""
```

#### `package.py`
```python
@dataclass
class Package:
    id: str
    destination: tuple[float, float]
    volume_m3: float
    payment: float
    priority: int = 1

    def __lt__(self, other: 'Package') -> bool:
        """For priority queue sorting"""
        return self.priority < other.priority
```

#### `route.py`
```python
@dataclass
class Route:
    vehicle: Vehicle
    packages: list[Package]
    stops: list[tuple[float, float]]

    @property
    def total_distance(self) -> float:
        """Calculate total route distance"""

    @property
    def total_cost(self) -> float:
        """Calculate total operating cost"""

    @property
    def total_revenue(self) -> float:
        """Sum of package payments"""

    @property
    def profit(self) -> float:
        """Revenue - Cost"""

    @property
    def total_volume(self) -> float:
        """Sum of package volumes"""

    def is_valid(self) -> bool:
        """Check all constraints satisfied"""
```

#### `map.py`
```python
class DeliveryMap:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.depot = (0, 0)
        self.locations: dict[str, tuple[float, float]] = {}

    def distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def manhattan_distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculate Manhattan distance (grid-based)"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
```

#### `game_state.py`
```python
@dataclass
class DayHistory:
    day: int
    packages_delivered: int
    revenue: float
    costs: float
    profit: float
    agent_used: str

class GameState:
    def __init__(self, initial_balance: float = 100000):
        self.current_day: int = 1
        self.balance: float = initial_balance
        self.fleet: list[Vehicle] = []
        self.packages_pending: list[Package] = []
        self.packages_delivered: list[Package] = []
        self.history: list[DayHistory] = []
        self.current_routes: list[Route] = []

    def advance_day(self) -> None:
        """Move to next day, load new packages"""

    def execute_routes(self) -> float:
        """Run planned routes, return profit"""

    def purchase_vehicle(self, vehicle_type: VehicleType) -> bool:
        """Buy new vehicle if affordable"""
```

---

### 2. Agents Layer (`src/agents/`)

#### Strategy Pattern Implementation

```python
from abc import ABC, abstractmethod

class RouteAgent(ABC):
    """Base class for all routing algorithms"""

    def __init__(self, delivery_map: DeliveryMap):
        self.map = delivery_map
        self.name = "Base Agent"

    @abstractmethod
    def plan_routes(self, packages: list[Package], fleet: list[Vehicle]) -> list[Route]:
        """
        Create optimized routes for given packages and fleet
        Returns: List of routes (one per vehicle used)
        """
        pass

    def calculate_metrics(self, routes: list[Route]) -> dict:
        """Calculate performance metrics for routes"""
        return {
            'total_distance': sum(r.total_distance for r in routes),
            'total_cost': sum(r.total_cost for r in routes),
            'total_revenue': sum(r.total_revenue for r in routes),
            'profit': sum(r.profit for r in routes),
            'vehicles_used': len(routes),
            'packages_delivered': sum(len(r.packages) for r in routes)
        }
```

#### `greedy_agent.py`
```python
class GreedyAgent(RouteAgent):
    """Nearest Neighbor + First Fit strategy"""

    def __init__(self, delivery_map: DeliveryMap):
        super().__init__(delivery_map)
        self.name = "Greedy Agent"

    def plan_routes(self, packages: list[Package], fleet: list[Vehicle]) -> list[Route]:
        """
        Algorithm:
        1. Sort packages by proximity to depot (greedy heuristic)
        2. For each package, assign to first vehicle with capacity
        3. Add to route using nearest neighbor insertion

        Time Complexity: O(n * m) where n=packages, m=vehicles
        Space Complexity: O(n)
        """
        routes = []
        # Sort packages by distance from depot (greedy choice)
        sorted_packages = sorted(packages,
                                key=lambda p: self.map.distance(self.map.depot, p.destination))

        # First-fit bin packing
        for pkg in sorted_packages:
            # Try to fit in existing route
            placed = False
            for route in routes:
                if route.vehicle.can_carry(route.total_volume + pkg.volume_m3):
                    route.packages.append(pkg)
                    placed = True
                    break

            # Need new vehicle
            if not placed and fleet:
                vehicle = fleet.pop(0)
                new_route = Route(vehicle=vehicle, packages=[pkg], stops=[])
                routes.append(new_route)

        # Optimize stop order for each route (nearest neighbor TSP)
        for route in routes:
            route.stops = self._optimize_stops(route.packages)

        return routes

    def _optimize_stops(self, packages: list[Package]) -> list[tuple[float, float]]:
        """Nearest neighbor TSP for stop order"""
        # Implementation here
        pass
```

#### `backtracking_agent.py`
```python
class BacktrackingAgent(RouteAgent):
    """Exhaustive search with pruning"""

    def __init__(self, delivery_map: DeliveryMap, max_packages: int = 20):
        super().__init__(delivery_map)
        self.name = "Backtracking Agent"
        self.max_packages = max_packages  # Limit for tractability
        self.best_solution = None
        self.best_profit = float('-inf')

    def plan_routes(self, packages: list[Package], fleet: list[Vehicle]) -> list[Route]:
        """
        Algorithm:
        1. Try all possible package-to-vehicle assignments
        2. Prune branches that violate capacity
        3. Keep track of best solution found

        Time Complexity: O(m^n) where m=vehicles, n=packages (with pruning)
        Space Complexity: O(n) for recursion stack
        """
        if len(packages) > self.max_packages:
            print(f"Warning: Too many packages ({len(packages)}), using first {self.max_packages}")
            packages = packages[:self.max_packages]

        self.best_solution = None
        self.best_profit = float('-inf')

        # Start recursive backtracking
        self._backtrack(packages, fleet, [], 0)

        return self.best_solution or []

    def _backtrack(self, packages: list[Package], fleet: list[Vehicle],
                   current_routes: list[Route], package_idx: int):
        """Recursive backtracking"""
        # Base case: all packages assigned
        if package_idx == len(packages):
            profit = sum(r.profit for r in current_routes)
            if profit > self.best_profit:
                self.best_profit = profit
                self.best_solution = [Route(r.vehicle, r.packages[:], [])
                                     for r in current_routes]
            return

        package = packages[package_idx]

        # Try adding to each existing route
        for route in current_routes:
            if route.vehicle.can_carry(route.total_volume + package.volume_m3):
                route.packages.append(package)
                self._backtrack(packages, fleet, current_routes, package_idx + 1)
                route.packages.pop()  # Backtrack

        # Try creating new route with available vehicle
        if fleet:
            vehicle = fleet.pop(0)
            new_route = Route(vehicle=vehicle, packages=[package], stops=[])
            current_routes.append(new_route)
            self._backtrack(packages, fleet, current_routes, package_idx + 1)
            current_routes.pop()  # Backtrack
            fleet.insert(0, vehicle)
```

---

### 3. Core Engine (`src/core/`)

#### `engine.py`
```python
class GameEngine:
    """Orchestrates game flow and state transitions"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.state = GameState()
        self.delivery_map = DeliveryMap(100, 100)
        self.agents: dict[str, RouteAgent] = {}
        self._register_agents()

    def _register_agents(self):
        """Initialize all available agents"""
        self.agents['greedy'] = GreedyAgent(self.delivery_map)
        self.agents['backtracking'] = BacktrackingAgent(self.delivery_map)
        # Add more agents as implemented

    def load_day_packages(self, day: int) -> list[Package]:
        """Load packages for specific day from JSON"""

    def test_agent(self, agent_name: str) -> dict:
        """Test an agent and return metrics without executing"""
        agent = self.agents[agent_name]
        routes = agent.plan_routes(self.state.packages_pending, self.state.fleet[:])
        return agent.calculate_metrics(routes)

    def apply_agent_solution(self, agent_name: str):
        """Apply agent's solution to game state"""
        agent = self.agents[agent_name]
        self.state.current_routes = agent.plan_routes(
            self.state.packages_pending,
            self.state.fleet[:]
        )

    def execute_day(self):
        """Run the day with current routes"""
        profit = self.state.execute_routes()
        self.state.balance += profit
        self.state.advance_day()
```

#### `router.py`
```python
class Router:
    """Utility functions for route calculations"""

    @staticmethod
    def calculate_route_distance(stops: list[tuple[float, float]],
                                 delivery_map: DeliveryMap) -> float:
        """Calculate total distance for ordered stops"""

    @staticmethod
    def tsp_nearest_neighbor(points: list[tuple[float, float]],
                            start: tuple[float, float],
                            delivery_map: DeliveryMap) -> list[tuple[float, float]]:
        """Solve TSP using nearest neighbor heuristic"""

    @staticmethod
    def tsp_2opt(route: list[tuple[float, float]],
                 delivery_map: DeliveryMap) -> list[tuple[float, float]]:
        """Improve route using 2-opt local search"""
```

---

### 4. UI Layer (`src/ui/`) - Pygame Based

#### `renderer.py`
```python
class MapRenderer:
    """Renders the delivery map and routes"""

    def __init__(self, screen: pygame.Surface, delivery_map: DeliveryMap):
        self.screen = screen
        self.map = delivery_map
        self.scale = 5  # pixels per km

    def draw_map(self):
        """Draw base map with depot"""

    def draw_package(self, package: Package, color: tuple):
        """Draw package destination as point"""

    def draw_route(self, route: Route, color: tuple, animate: bool = False):
        """Draw route line with vehicle path"""

    def draw_vehicle(self, vehicle: Vehicle, position: tuple):
        """Draw vehicle icon at position"""
```

#### `ui_components.py`
```python
class StatsPanel:
    """Display game statistics"""
    def render(self, screen: pygame.Surface, state: GameState):
        pass

class ControlPanel:
    """Buttons and controls for game actions"""
    def render(self, screen: pygame.Surface):
        pass

class AgentSelector:
    """UI for selecting and comparing agents"""
    def render(self, screen: pygame.Surface, agents: dict):
        pass
```

---

### 5. Data Layer (`src/utils/`)

#### `data_loader.py`
```python
class DataLoader:
    """Handles loading from JSON files"""

    @staticmethod
    def load_vehicle_types(filepath: Path) -> dict[str, VehicleType]:
        """Load vehicle type definitions"""

    @staticmethod
    def load_packages(filepath: Path) -> list[Package]:
        """Load packages for a day"""

    @staticmethod
    def load_map(filepath: Path) -> DeliveryMap:
        """Load map configuration"""

    @staticmethod
    def load_game_state(filepath: Path) -> GameState:
        """Load saved game"""
```

---

## Key Design Patterns

### 1. Strategy Pattern (Agents)
- Different algorithms implement same interface
- Easy to add new agents
- Runtime algorithm selection

### 2. Data Class Pattern (Models)
- Immutable where possible
- Auto-generated methods (__init__, __repr__)
- Type safety

### 3. Dependency Injection
- Map passed to agents
- Testable components
- Loose coupling

### 4. Repository Pattern (Data Layer)
- Abstracted data access
- Easy to swap storage (JSON → Database)

---

## Performance Considerations

### Algorithm Complexity Targets
- Greedy: O(n log n) - Fast for any size
- Backtracking: Limited to ~20 packages (exponential)
- DP: O(n²) - Medium datasets
- Genetic: O(generations × population × n) - Tunable

### Optimization Techniques
1. **Caching:** Memoize distance calculations
2. **Pruning:** Early termination in backtracking
3. **Heuristics:** Guide search toward good solutions
4. **Batching:** Process packages in chunks if needed

---

## Testing Strategy

### Unit Tests
- Each model method
- Distance calculations
- Constraint validators
- Agent correctness

### Integration Tests
- Full day simulation
- Agent comparisons
- Save/load functionality

### Test Data
- Small dataset (5 packages, 2 vehicles) - quick validation
- Medium dataset (50 packages, 5 vehicles) - realistic
- Large dataset (200 packages, 10 vehicles) - stress test

---

## Next: Implementation Phase 1

Focus on:
1. Core models with type hints
2. JSON data structures
3. Basic validation
4. Console-based interaction
5. Manual route creation

This foundation enables rapid development of algorithms and UI in later phases.
