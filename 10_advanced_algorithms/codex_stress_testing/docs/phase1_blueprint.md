# Phase 1 Blueprint – Delivery Fleet Management System

## Objectives
- Stand up a reusable codebase foundation that mirrors the specification’s core entities (vehicles, packages, routes, map, game state).
- Enable JSON-driven configuration and save/load workflows for deterministic simulations.
- Provide a CLI-based admin loop that supports package intake, manual assignment scaffolding, validation, and financial bookkeeping.
- Instrument the code to attach early metrics/totals that later agents and the UI can reuse.

## Map & World Modeling
- Adopt a 2D Cartesian map (`float` km coordinates) with configurable bounds and named points of interest.
- Distance abstraction exposes both Euclidean and Manhattan metrics; default to Euclidean with strategy flag for experimentation.
- Packages originate at depot `(0, 0)`, but maintain explicit `origin` field to future-proof multi-depot scenarios.
- Persist map and landmarks in `data/map.json`; provide loader validation against declared bounds.

## Data & Persistence
- JSON schema per spec (`vehicles.json`, `packages_dayX.json`, `savegame.json`).
- Loader service to hydrate dataclasses with type checking and graceful fallbacks (e.g., unknown vehicle type → warning).
- Save system writes to `savegames/` to avoid clobbering seed data; include timestamped filenames for history.
- Config module centralizes defaults (starting balance, daily package file naming pattern, cost toggles).

## Core Models & Services
- Dataclasses for `Vehicle`, `Package`, `Route`, `Map`, `GameState`, `DailySummary`, and `Assignment`.
- `InventoryManager` handles fleet CRUD (purchase, availability, range checks).
- `PackageQueue` abstracts pending/assigned/in-transit transitions.
- `RouteBuilder` and `RouteValidator` manage constraint enforcement (capacity, depot return, total distance vs max range).
- Financial engine encapsulates income/expense posting with ledger entries for audit trail.

## CLI Planning Loop (Phase 1)
- Menu-driven console flow:
  1. Load day’s packages.
  2. Inspect fleet and pending packages (filtered, sorted views).
  3. Assign packages to vehicles manually (basic prompts; algorithm hooks stubbed for Phase 2).
  4. Validate routes and compute projected metrics.
  5. Run simulated execution to update balance and package statuses.
- Output summarized tables (distance, cost, profit) to establish consistent format for later agent comparisons.

## Extensibility Hooks
- Strategy pattern for routing agents: `BaseAgent` abstract class returning `RoutePlan`.
- Observer-style event dispatch for UI/visualization integration (Phase 3) without refactoring core logic.
- Metrics service to compute KPI snapshots (utilization, profit per km, delivery rate).
- Configuration-based toggles (e.g., enabling weight constraints, range limits) to support advanced lessons.

## Testing Approach
- Unit tests for loaders, validators, and financial calculations using `pytest`.
- Fixture JSON files in `tests/fixtures/` to ensure deterministic expectations.
- Smoke test simulating a single day from package intake to balance update.

## Risks & Mitigations
- **Complexity creep:** Lock Phase 1 scope to manual assignment + validation; defer automation to Phase 2.
- **Data integrity:** Enforce schema validation and descriptive errors when loading malformed JSON.
- **Performance:** Use lazy evaluation for distance matrix caching but gate behind optional flag until needed.

## Deliverables for Phase 1
- Source tree scaffold with core modules and dataclasses.
- JSON loader utilities and seed data for map, vehicles, and at least two daily package manifests.
- Console-based day cycle with manual assignment placeholders and financial updates.
- Initial pytest suite covering loaders, validators, and financial ledger.
