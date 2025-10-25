# Delivery Fleet Management System - Specifications

## Project Overview
An interactive delivery company management simulation where students learn algorithmic optimization through route planning, fleet management, and financial decision-making.

---

## 1. Core Entities

### 1.1 Vehicle
**Attributes:**
- `id`: Unique identifier
- `type`: Category (e.g., "small_van", "medium_truck", "large_truck")
- `capacity_m3`: Maximum carrying volume in cubic meters
- `cost_per_km`: Operating cost per kilometer
- `purchase_price`: Initial acquisition cost
- `max_range_km`: Maximum distance per day (optional constraint)
- `current_location`: Current position on map (x, y coordinates)

**Vehicle Types (Suggested defaults):**
```json
{
  "small_van": {"capacity_m3": 10, "cost_per_km": 0.50, "purchase_price": 15000},
  "medium_truck": {"capacity_m3": 25, "cost_per_km": 0.80, "purchase_price": 35000},
  "large_truck": {"capacity_m3": 50, "cost_per_km": 1.20, "purchase_price": 65000}
}
```

### 1.2 Package
**Attributes:**
- `id`: Unique identifier
- `origin`: Company home address (0, 0) - depot coordinates
- `destination`: Delivery address (x, y coordinates)
- `volume_m3`: Package size in cubic meters
- `payment_received`: Revenue for delivery
- `weight_kg`: Weight (optional, for future extensions)
- `priority`: Delivery priority (1-5, optional)
- `received_date`: Day package was received
- `delivery_deadline`: Day by which it must be delivered (optional)

### 1.3 Map/Location
**Representation:**
- 2D Cartesian coordinate system
- Depot at origin (0, 0)
- Delivery points scattered across grid
- Distance calculation: Euclidean or Manhattan distance

**Map Structure:**
- `width`: Map width in km
- `height`: Map height in km
- `locations`: List of named locations with coordinates
- `depot`: Company home base coordinates (0, 0)

### 1.4 Route
**Attributes:**
- `vehicle_id`: Assigned vehicle
- `package_ids`: List of packages in delivery order
- `stops`: Ordered list of coordinates to visit
- `total_distance_km`: Calculated route distance
- `total_cost`: Operating cost for this route
- `total_revenue`: Sum of package payments
- `total_volume`: Sum of package volumes
- `is_valid`: Whether route satisfies all constraints

### 1.5 Game State
**Attributes:**
- `current_day`: Current simulation day
- `balance`: Company financial balance
- `fleet`: List of owned vehicles
- `packages_pending`: Packages awaiting assignment
- `packages_in_transit`: Packages currently being delivered
- `packages_delivered`: Completed deliveries
- `daily_history`: Performance metrics per day

---

## 2. Game Mechanics

### 2.1 Day Cycle
**Phase 1: Planning (00:00 - Start of Day)**
1. Receive list of packages for the day
2. Review available fleet
3. Test different routing algorithms (agents)
4. Assign packages to vehicles and create routes
5. Review predictions (cost, revenue, efficiency)

**Phase 2: Execution (Run the Day)**
1. Execute planned routes
2. Vehicles travel and deliver packages
3. Calculate actual costs and revenues
4. Update balance and statistics

**Phase 3: End of Day**
1. Review performance metrics
2. Option to purchase new vehicles
3. Advance to next day

### 2.2 Financial System
**Income:**
- Package delivery payments
- Payment received upon successful delivery

**Expenses:**
- Vehicle operating costs (distance × cost_per_km)
- Vehicle purchase costs
- Maintenance costs (optional, future)

**Balance Tracking:**
- Starting balance: $100,000 (configurable)
- Daily profit/loss calculation
- Win/lose conditions based on balance

### 2.3 Constraints & Rules
**Hard Constraints (Must satisfy):**
1. Package volume ≤ Vehicle capacity
2. All packages assigned to a vehicle
3. Each package delivered exactly once
4. Routes must start and end at depot

**Soft Constraints (Optimization goals):**
1. Minimize total distance
2. Maximize profit (revenue - costs)
3. Minimize number of vehicles used
4. Minimize route overlap

---

## 3. AI Agents (Algorithms)

Students will implement these as routing optimization agents:

### 3.1 Greedy Algorithm
**Strategy:** Make locally optimal choice at each step
**Implementation:**
- Nearest neighbor first
- First-fit bin packing for vehicle loading
- Pros: Fast, simple
- Cons: May not find global optimum

### 3.2 Backtracking
**Strategy:** Explore all possibilities with pruning
**Implementation:**
- Try all package-to-vehicle assignments
- Prune branches that violate constraints
- Find all valid solutions
- Pros: Complete solution space exploration
- Cons: Can be slow for large problems

### 3.3 Dynamic Programming
**Strategy:** Build optimal solution from subproblems
**Implementation:**
- Subset-sum for capacity optimization
- Traveling Salesman Problem (TSP) for route order
- Pros: Optimal solutions for specific subproblems
- Cons: Memory intensive

### 3.4 Genetic Algorithm (Advanced)
**Strategy:** Evolutionary approach
**Implementation:**
- Population of route solutions
- Fitness function: profit, distance
- Crossover and mutation operators
- Pros: Good for complex optimization
- Cons: Non-deterministic, parameter tuning needed

### 3.5 A* Search (Advanced)
**Strategy:** Informed search with heuristics
**Implementation:**
- Heuristic: straight-line distance to destinations
- Find shortest paths on map
- Pros: Optimal with admissible heuristic
- Cons: Memory requirements

### 3.6 Custom/Manual
**Strategy:** Admin manually creates routes
**Implementation:**
- Interactive assignment interface
- Manual package-to-vehicle assignment
- Pros: Full control, learning tool
- Cons: Time-consuming, human error

---

## 4. Technical Stack

### 4.1 Recommended Architecture

**Core (Python):**
- **Language:** Python 3.10+
- **OOP Design:** Classes for Vehicle, Package, Route, Map, Agent, GameState
- **Data Persistence:** JSON files for configuration and save states

**Visualization:**
- **Option A (Recommended):** Pygame
  - Good 2D graphics
  - Event handling for interactivity
  - Easy to draw maps, routes, vehicles
  - Student-friendly

- **Option B:** Web-based (Flask + HTML Canvas/JavaScript)
  - Modern UI
  - Better for complex visualizations
  - Requires web dev knowledge

**Libraries:**
- `pygame` - Graphics and UI
- `json` - Data persistence
- `dataclasses` - Clean data models
- `typing` - Type hints
- `matplotlib` - (Optional) Analytics charts
- `numpy` - (Optional) Numerical operations

### 4.2 Project Structure
```
delivery_fleet_game/
├── data/
│   ├── vehicles.json          # Vehicle type definitions
│   ├── map.json               # Map configuration
│   ├── packages_day1.json     # Package data per day
│   ├── packages_day2.json
│   └── savegame.json          # Game state persistence
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vehicle.py
│   │   ├── package.py
│   │   ├── route.py
│   │   ├── map.py
│   │   └── game_state.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py
│   │   ├── greedy_agent.py
│   │   ├── backtracking_agent.py
│   │   ├── dp_agent.py
│   │   └── genetic_agent.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── renderer.py
│   │   ├── map_view.py
│   │   └── ui_components.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── engine.py          # Game loop and logic
│   │   ├── router.py          # Route calculation utilities
│   │   └── validator.py       # Constraint checking
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       └── metrics.py
├── tests/
│   ├── test_agents.py
│   ├── test_models.py
│   └── test_routing.py
├── main.py                     # Entry point
├── requirements.txt
└── README.md
```

---

## 5. Data Schemas (JSON)

### 5.1 vehicles.json
```json
{
  "vehicle_types": {
    "small_van": {
      "capacity_m3": 10,
      "cost_per_km": 0.50,
      "purchase_price": 15000,
      "max_range_km": 200
    },
    "medium_truck": {
      "capacity_m3": 25,
      "cost_per_km": 0.80,
      "purchase_price": 35000,
      "max_range_km": 300
    },
    "large_truck": {
      "capacity_m3": 50,
      "cost_per_km": 1.20,
      "purchase_price": 65000,
      "max_range_km": 400
    }
  }
}
```

### 5.2 map.json
```json
{
  "width": 100,
  "height": 100,
  "depot": {"x": 0, "y": 0},
  "locations": [
    {"id": "loc_1", "name": "Residential Area A", "x": 20, "y": 30},
    {"id": "loc_2", "name": "Business District", "x": 45, "y": 60},
    {"id": "loc_3", "name": "Suburbs", "x": 70, "y": 20}
  ]
}
```

### 5.3 packages_day1.json
```json
{
  "day": 1,
  "packages": [
    {
      "id": "pkg_001",
      "destination": {"x": 20, "y": 30},
      "volume_m3": 2.5,
      "payment": 50,
      "priority": 1
    },
    {
      "id": "pkg_002",
      "destination": {"x": 45, "y": 60},
      "volume_m3": 5.0,
      "payment": 80,
      "priority": 2
    }
  ]
}
```

### 5.4 savegame.json
```json
{
  "current_day": 3,
  "balance": 125000,
  "fleet": [
    {
      "id": "veh_001",
      "type": "small_van",
      "purchase_day": 1
    }
  ],
  "history": [
    {
      "day": 1,
      "packages_delivered": 15,
      "revenue": 1200,
      "costs": 400,
      "profit": 800
    }
  ]
}
```

---

## 6. Three-Phase Implementation Plan

### **PHASE 1: Foundation & Core Models** (Week 1-2)
**Goal:** Establish data structures and basic game mechanics

**Deliverables:**
1. ✅ Core data models (Vehicle, Package, Route, Map, GameState)
2. ✅ JSON data loading system
3. ✅ Basic game engine (day cycle, state management)
4. ✅ Simple console-based UI (text output)
5. ✅ Manual route creation (admin assigns packages)
6. ✅ Route validation (constraint checking)
7. ✅ Basic financial calculations

**Test Case:**
- Load 10 packages, 2 vehicles
- Manually assign packages to vehicles
- Calculate route distance and cost
- Update balance

---

### **PHASE 2: AI Agents & Optimization** (Week 3-4)
**Goal:** Implement algorithmic route planning

**Deliverables:**
1. ✅ Base Agent abstract class
2. ✅ Greedy Agent implementation
3. ✅ Backtracking Agent implementation
4. ✅ Agent comparison system
5. ✅ Route optimization metrics (distance, cost, profit)
6. ✅ Enhanced console output with agent stats
7. ✅ Unit tests for agents

**Test Case:**
- Load 50 packages, 5 vehicles
- Run Greedy vs Backtracking agents
- Compare results (time, profit, distance)
- Validate constraint satisfaction

---

### **PHASE 3: Visualization & Polish** (Week 5-6)
**Goal:** Create interactive graphical interface

**Deliverables:**
1. ✅ Pygame-based map visualization
2. ✅ Interactive route display (animated)
3. ✅ UI panels (stats, controls, agent selection)
4. ✅ Visual comparison of agent solutions
5. ✅ Fleet management interface (buy vehicles)
6. ✅ Save/load game functionality
7. ✅ Performance metrics dashboard
8. ✅ Additional agents (DP, Genetic - optional)

**Test Case:**
- Play full game: 7 days
- Use different agents each day
- Purchase new vehicles
- Visualize routes on map
- Track balance over time

---

## 7. Key Features for "Art of Programming"

### 7.1 Algorithm Visualization
- Show step-by-step how each agent makes decisions
- Highlight differences between greedy and optimal approaches
- Display algorithm time complexity in practice

### 7.2 Comparative Analysis
- Side-by-side agent comparisons
- Metrics: runtime, solution quality, consistency
- Trade-off discussions (speed vs optimality)

### 7.3 Extensibility
- Easy to add new agent types
- Pluggable architecture
- Students can implement custom agents

### 7.4 Educational Value
- Comments explaining algorithmic choices
- Design pattern examples (Strategy, Factory)
- Clean code principles

---

## 8. Win/Lose Conditions

**Win Scenarios:**
- Reach day 30 with balance > $200,000
- Achieve 90%+ delivery success rate
- Maintain positive daily profit for 10 consecutive days

**Lose Scenarios:**
- Balance drops below $0 (bankruptcy)
- Fail to deliver 50%+ of packages in a day
- Three consecutive days of losses

---

## 9. Future Extensions (Post-MVP)

1. **Multi-day planning:** Packages can wait for next day
2. **Time windows:** Deliveries must occur within specific hours
3. **Traffic simulation:** Dynamic route costs
4. **Vehicle breakdowns:** Random events
5. **Customer satisfaction:** Ratings based on delivery speed
6. **Competitor AI:** Rival delivery companies
7. **Multiplayer mode:** Students compete for best strategy

---

## 10. Success Metrics

**For Students:**
- Understand algorithm trade-offs (greedy vs optimal)
- Practice OOP design and clean architecture
- Learn data persistence and state management
- Gain experience with visualization libraries
- Apply algorithmic thinking to real-world problems

**For Game:**
- Engaging and educational gameplay
- Clear visualization of algorithmic concepts
- Balanced difficulty curve
- Replayability through different strategies

---

## Next Steps
1. Review and approve specification
2. Set up project structure
3. Begin Phase 1 implementation
4. Create initial data files
5. Build core models and game engine

---

**Version:** 1.0
**Author:** Claude Code
**Date:** 2025-10-24
**Course:** Art of Programming - Advanced Algorithms
